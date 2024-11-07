import multiprocessing
import contextlib
from sys import getsizeof
from collections import defaultdict
import ctypes

import numpy as np
import torch

from distributed_optimization_library.compressor import CompressedVector, CompressedTorchVector, \
    SumCompressedVectors, CoreCompressedValues, CompressedNatVector, GZipVector
from distributed_optimization_library.function import FunctionType

NUMBER_OF_BITES_IN_BYTE = 8


class PartialParticipationError(object):
    pass


def estimate_size_of_object(obj):
    type_of_object = type(obj)
    if isinstance(obj, np.ndarray):
        assert obj.ndim <= 1
        if obj.ndim == 1:
            return len(obj) * obj.itemsize * NUMBER_OF_BITES_IN_BYTE
        else:
            return obj.itemsize * NUMBER_OF_BITES_IN_BYTE
    elif isinstance(obj, (list, tuple)):
        total_size = 0
        for el in obj:
            total_size += estimate_size_of_object(el)
        return total_size
    elif isinstance(obj, np.floating):
        return obj.itemsize * NUMBER_OF_BITES_IN_BYTE
    elif isinstance(obj, (bool)):
        return 1
    elif isinstance(obj, (float, int, np.int64)):
        return 64
    elif isinstance(obj, str):
        return getsizeof(obj)
    elif isinstance(obj, (CompressedVector, 
                          CompressedTorchVector,
                          SumCompressedVectors,
                          CoreCompressedValues,
                          CompressedNatVector,
                          GZipVector)):
        return obj.size_in_memory()
    elif isinstance(obj, BroadcastNumpy):
        return obj.size_in_memory()
    elif torch.is_tensor(obj):
        assert obj.ndim <= 1
        if obj.dtype == torch.float32:
            itemsize = 4
        elif obj.dtype == torch.float64:
            itemsize = 8
        else:
            raise RuntimeError()
        if obj.ndim == 1:
            return len(obj) * itemsize * NUMBER_OF_BITES_IN_BYTE
        else:
            return itemsize * NUMBER_OF_BITES_IN_BYTE
    elif obj is None:
        return NUMBER_OF_BITES_IN_BYTE
    elif isinstance(obj, FunctionType):
        return NUMBER_OF_BITES_IN_BYTE
    elif isinstance(obj, PartialParticipationError):
        return 0

    raise RuntimeError("Wrong type of an object: {}".format(type_of_object))


class BroadcastNumpy(object):
    _LABEL = '_SHARED_MEMORY'
    def __init__(self, arr):
        assert arr.dtype == np.float32
        self._arr = arr
    
    def size_in_memory(self):
        return estimate_size_of_object(self._arr)
    
    @staticmethod
    def _copy_to_shared_memory(kwargs, shared_memory):
        found_number = 0
        shared_memory_info = []
        keys = list(kwargs.keys())
        for k in keys:
            if isinstance(kwargs[k], BroadcastNumpy):
                assert found_number < len(shared_memory)
                broadcast_numpy = kwargs[k]
                count = len(broadcast_numpy._arr)
                assert len(shared_memory[found_number]) >= count, "Allocate more shared memory"
                shared_memory[found_number][:count] = broadcast_numpy._arr
                del kwargs[k]
                shared_memory_info.append((k, found_number, count))
                found_number += 1
        kwargs[BroadcastNumpy._LABEL] = shared_memory_info
    
    @staticmethod
    def _ignore_broadcast(kwargs):
        for k in kwargs:
            if isinstance(kwargs[k], BroadcastNumpy):
                kwargs[k] = kwargs[k]._arr


def find_total_node(stat_node):
    total = 0
    for node_method in stat_node:
        total += stat_node[node_method]
    return total


def find_total(stat_nodes):
    total = 0
    for stat_node in stat_nodes:
        total += find_total_node(stat_node)
    return total


def find_max(stat_nodes):
    max_ = float('-inf')
    for node in stat_nodes:
        total = 0
        for node_method in node:
            total += node[node_method]
        max_ = max(max_, total)
    return max_


def node_parallel(node_signatures, conn, shared_memory):
    nodes = {}
    for node_index in node_signatures:
        nodes[node_index] = node_signatures[node_index].create_instance()
    while True:
        (node_indices, node_method, kwargs) = conn.recv()
        if BroadcastNumpy._LABEL in kwargs:
            for name, shared_memory_index, count in kwargs[BroadcastNumpy._LABEL]:
                assert name not in kwargs
                kwargs[name] = np.copy(np.frombuffer(shared_memory[shared_memory_index], 
                                                     dtype=np.float32, count=count))
            del kwargs[BroadcastNumpy._LABEL]
        assert isinstance(node_indices, list)
        outputs = []
        for node_index in node_indices:
            output = getattr(nodes[node_index], node_method)(**kwargs)
            outputs.append((node_index, output))
        conn.send(outputs)


class Transport(object):
    def __init__(self, nodes, parallel=False, shared_memory_size=0, shared_memory_len=1, number_of_processes=1):
        self._nodes = nodes
        self._parallel = parallel
        self._number_of_nodes = len(nodes)
        self._number_of_processes = min(self._number_of_nodes, number_of_processes)
        self._ignore_statistics = False
        self._stat_to_nodes = [defaultdict(int) 
                               for _ in range(len(self._nodes))]
        self._stat_from_nodes = [defaultdict(int) 
                                 for _ in range(len(self._nodes))]
        self._max_stat_from_nodes = defaultdict(int)
        if self._parallel:
            assert torch.multiprocessing.get_sharing_strategy() == "file_system"
            self._connections = []
            self._processes = []
            self._shared_memory = [multiprocessing.Array(
                ctypes.c_float, shared_memory_size, lock=False) for _ in range(shared_memory_len)]
            self._shared_memory_numpy = [np.frombuffer(shared_memory, dtype=np.float32)
                                         for shared_memory in self._shared_memory]
            self._groups = [{} for _ in range(self._number_of_processes)]
            for node_index in range(len(self._nodes)):
                group_index = self._group_index(node_index)
                self._groups[group_index][node_index] = self._nodes[node_index]
            for process_index in range(self._number_of_processes):
                parent_conn, child_conn = torch.multiprocessing.Pipe()
                self._connections.append(parent_conn)   
                process = torch.multiprocessing.Process(
                    target=node_parallel, args=(self._groups[process_index], 
                                                child_conn, self._shared_memory))
                process.start()
                self._processes.append(process)
        else:
            self._nodes = [node.create_instance() for node in self._nodes]
    
    def get_number_of_nodes(self):
        return len(self._nodes)
    
    def call_nodes_method(self, node_method, node_indices=None, **kwargs):
        node_indices = node_indices if node_indices is not None else list(range(self._number_of_nodes))
        if not self._parallel:
            outputs = []
            self._init_max_stat_current()
            for node_index in node_indices:
                outputs.append(self._call_node_method(node_index, node_method, **kwargs))
            self._aggregate_max_stat_current()
            return outputs
        else:
            return self._call_nodes_method_parallel(node_method, node_indices, **kwargs)
    
    def call_node_method(self, node_index, node_method, **kwargs):
        self._init_max_stat_current()
        outputs = self._call_node_method(node_index, node_method, **kwargs)
        self._aggregate_max_stat_current()
        return outputs
        
    def _call_node_method(self, node_index, node_method, **kwargs):
        self._update_to_nodes_stat(node_index, node_method, kwargs)
        BroadcastNumpy._ignore_broadcast(kwargs)
        if not self._parallel:
            outputs = getattr(self._nodes[node_index], node_method)(**kwargs)
        else:
            group_index = self._group_index(node_index)
            self._connections[group_index].send([[node_index], node_method, kwargs])
            outputs = self._connections[group_index].recv()[0][1]
        self._update_from_nodes_stat(node_index, node_method, outputs)
        return outputs
    
    def get_stat_from_nodes(self):
        return [dict(d) for d in self._stat_from_nodes]
    
    def get_max_stat_from_nodes(self):
        return dict(self._max_stat_from_nodes)
    
    def get_stat_to_nodes(self):
        return [dict(d) for d in self._stat_to_nodes]
    
    def stop(self):
        if self._parallel:
            for process in self._processes:
                process.terminate()
    
    def _group_index(self, node_index):
        return node_index % self._number_of_processes

    def _update_to_nodes_stat(self, node_index, node_method, kwargs):
        for arg_name in kwargs:
            arg = kwargs[arg_name]
            self._update_nodes_stat(self._stat_to_nodes, node_index, node_method, arg,
                                    max_stat=None)
    
    def _update_from_nodes_stat(self, node_index, node_method, outputs):
        if outputs is None:
            return
        self._update_nodes_stat(self._stat_from_nodes, node_index, node_method, outputs,
                                max_stat=self._max_stat_from_nodes_current)
    
    def _update_nodes_stat(self, stat, node_index, node_method, object, max_stat):
        if not self._ignore_statistics:
            size_of_object = estimate_size_of_object(object)
            stat[node_index][node_method] += size_of_object
            if max_stat is not None:
                max_stat[node_method] = max(max_stat[node_method], size_of_object)
    
    def _call_nodes_method_parallel(self, node_method, node_indices, **kwargs):
        nodes_indices_per_group = [[] for _ in range(self._number_of_processes)]
        node_index_to_order_index = {}
        for order_index, node_index in enumerate(node_indices):
            self._update_to_nodes_stat(node_index, node_method, kwargs)
            group_index = self._group_index(node_index)
            nodes_indices_per_group[group_index].append(node_index)
            node_index_to_order_index[node_index] = order_index
        BroadcastNumpy._copy_to_shared_memory(kwargs, self._shared_memory_numpy)
        for group_index in range(self._number_of_processes):
            self._connections[group_index].send(
                [nodes_indices_per_group[group_index], node_method, kwargs])
        outputs = [None] * len(node_indices)
        self._init_max_stat_current()
        for group_index in range(self._number_of_processes):
            node_index_outputs = self._connections[group_index].recv()
            for (node_index, output) in node_index_outputs:
                self._update_from_nodes_stat(node_index, node_method, output)
                order_index = node_index_to_order_index[node_index]
                assert outputs[order_index] is None
                outputs[order_index] = output
        self._aggregate_max_stat_current()
        return outputs

    def _init_max_stat_current(self):
        self._max_stat_from_nodes_current = defaultdict(int)
        
    def _aggregate_max_stat_current(self):
        for node_method in self._max_stat_from_nodes_current:
            self._max_stat_from_nodes[node_method] += self._max_stat_from_nodes_current[node_method]
        self._max_stat_from_nodes_current = None
    
    @contextlib.contextmanager
    def ignore_statistics(self):
        assert not self._ignore_statistics
        self._ignore_statistics = True
        yield
        self._ignore_statistics = False
