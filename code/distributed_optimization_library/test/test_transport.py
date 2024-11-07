import numpy as np
import torch
import pytest

from distributed_optimization_library.transport import Transport, find_total, find_max
from distributed_optimization_library.compressor import CompressedVector, CompressedTorchVector
from distributed_optimization_library.signature import Signature


class DummyNode():
    
    def do_nothing(self, foo_arr, baz_arr):
        return

    def return_array(self):
        output_arr = np.zeros((13,), dtype=np.float64)
        output_arr[:4] = 3.14
        return output_arr


def test_transport():
    transport = Transport([Signature(DummyNode)])
    foo_arr = np.zeros((54,), dtype=np.float32)
    foo_arr[1:7] = 3.14
    baz_arr = np.zeros((11,), dtype=np.float64)
    baz_arr[:2] = 3.14
    transport.call_node_method(node_index=0,
                               node_method="do_nothing",
                               foo_arr=foo_arr,
                               baz_arr=baz_arr) # 54 * 32 + 11 * 64
    compressed_foo_arr = CompressedVector(
        range(1, 7), np.array([3.14] * 6, dtype=np.float32), 54)
    assert compressed_foo_arr.decompress().tolist() == foo_arr.tolist()
    compresse_torch_foo_arr = CompressedTorchVector(
        torch.tensor(range(2, 7)), torch.tensor(np.array([3.14] * 5, dtype=np.float32)), 43)
    transport.call_node_method(node_index=0,
                               node_method="do_nothing",
                               foo_arr=compressed_foo_arr,
                               baz_arr=compresse_torch_foo_arr) # 6 * 32 + 5 * 32
    foo_arr_torch = torch.tensor(np.zeros((9,), dtype=np.float32))
    foo_arr[1:7] = 3.14
    baz_arr_torch = torch.tensor(np.zeros((7,), dtype=np.float64))
    baz_arr[:2] = 3.14
    transport.call_node_method(node_index=0,
                               node_method="do_nothing",
                               foo_arr=foo_arr_torch,
                               baz_arr=baz_arr_torch) # 9 * 32 + 7 * 64
    
    transport.call_node_method(node_index=0,
                               node_method="return_array") # 13 * 64
    transport.call_nodes_method(node_method="return_array") # 13 * 64
    
    with transport.ignore_statistics():
        transport.call_node_method(node_index=0,
                                   node_method="do_nothing",
                                   foo_arr=foo_arr_torch,
                                   baz_arr=baz_arr_torch)
        transport.call_node_method(node_index=0, node_method="return_array")
    
    stat_from_nodes = transport.get_stat_from_nodes()
    stat_to_nodes = transport.get_stat_to_nodes()
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    
    assert stat_to_nodes[0]["do_nothing"] == 54 * 32 + 11 * 64 + 6 * 32 + 5 * 32 + 9 * 32 + 7 * 64
    assert stat_from_nodes[0]["return_array"] == 13 * 64 + 13 * 64
    assert max_stat_from_nodes["return_array"] == stat_from_nodes[0]["return_array"]


class DummyNodeForCallNodes():
    
    def __init__(self, index):
        self._index = index

    def return_array(self, foo_arr):
        output_arr = np.zeros((13,), dtype=np.int32)
        output_arr[0] = self._index
        return output_arr


torch.multiprocessing.set_sharing_strategy("file_system")


@pytest.mark.parametrize("parallel", [False, True])
def test_transport_call_nodes_methods_indices(parallel):
    transport = Transport([Signature(DummyNodeForCallNodes, index=0), 
                           Signature(DummyNodeForCallNodes, index=1)],
                          parallel=parallel,
                          number_of_processes=2)
    foo_arr = np.zeros((54,), dtype=np.float32)
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[0, 1],
                                          foo_arr=foo_arr)  # 54 * 32
    assert len(outputs) == 2
    assert outputs[0][0] == 0 and outputs[1][0] == 1
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[1, 0],
                                          foo_arr=foo_arr)  # 54 * 32
    assert len(outputs) == 2
    assert outputs[0][0] == 1 and outputs[1][0] == 0
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[1],
                                          foo_arr=foo_arr)  # 54 * 32
    assert len(outputs) == 1
    assert outputs[0][0] == 1
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[],
                                          foo_arr=foo_arr)
    assert len(outputs) == 0

    stat_to_nodes = transport.get_stat_to_nodes()
    assert stat_to_nodes[0]["return_array"] == 54 * 32 + 54 * 32
    assert stat_to_nodes[1]["return_array"] == 54 * 32 + 54 * 32 + 54 * 32
    
    stat_from_nodes = transport.get_stat_from_nodes()
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    assert stat_from_nodes[0]["return_array"] == 13 * 32 + 13 * 32
    assert stat_from_nodes[1]["return_array"] == 13 * 32 + 13 * 32 + 13 * 32
    assert max_stat_from_nodes["return_array"] == stat_from_nodes[1]["return_array"]
    transport.stop()


@pytest.mark.parametrize("parallel", [False, True])
def test_transport_call_nodes_methods_indices_empty(parallel):
    transport = Transport([Signature(DummyNodeForCallNodes, index=0), 
                           Signature(DummyNodeForCallNodes, index=1)],
                          parallel=parallel,
                          number_of_processes=2)
    foo_arr = np.zeros((54,), dtype=np.float32)
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[],
                                          foo_arr=foo_arr)
    assert len(outputs) == 0
    
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    assert max_stat_from_nodes == {}
    transport.stop()


@pytest.mark.parametrize("parallel", [False, True])
def test_transport_call_nodes_methods_indices_call_one(parallel):
    transport = Transport([Signature(DummyNodeForCallNodes, index=0), 
                           Signature(DummyNodeForCallNodes, index=1)],
                          parallel=parallel,
                          number_of_processes=2)
    foo_arr = np.zeros((54,), dtype=np.float32)
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[1],
                                          foo_arr=foo_arr)
    assert len(outputs) == 1
    assert outputs[0][0] == 1
    
    stat_to_nodes = transport.get_stat_to_nodes()
    assert stat_to_nodes[1]["return_array"] == 54 * 32  
    
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    stat_from_nodes = transport.get_stat_from_nodes()
    assert stat_from_nodes[1]["return_array"] == 13 * 32
    assert max_stat_from_nodes["return_array"] == stat_from_nodes[1]["return_array"]
    transport.stop()


@pytest.mark.parametrize("number_of_processes", [1, 3, 13])
def test_transport_call_nodes_methods_parallel_number_of_processes(number_of_processes):
    nodes = [Signature(DummyNodeForCallNodes, index=index) for index in range(1000)]
    transport = Transport(nodes,
                          parallel=True,
                          number_of_processes=number_of_processes)
    foo_arr = np.zeros((54,), dtype=np.float32)
    
    node_indices = [1, 999, 134]
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=node_indices,
                                          foo_arr=foo_arr)
    assert len(outputs) == 3
    assert outputs[0][0] == 1
    assert outputs[1][0] == 999
    assert outputs[2][0] == 134
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          node_indices=[],
                                          foo_arr=foo_arr)
    assert len(outputs) == 0
    
    outputs = transport.call_nodes_method(node_method="return_array",
                                          foo_arr=foo_arr)
    assert len(outputs) == 1000
    
    stat_to_nodes = transport.get_stat_to_nodes()
    for index in node_indices:
        assert stat_to_nodes[index]["return_array"] == 54 * 32 * 2
    
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    stat_from_nodes = transport.get_stat_from_nodes()
    for index in node_indices:
        assert stat_from_nodes[index]["return_array"] == 13 * 32 * 2
        assert max_stat_from_nodes["return_array"] == stat_from_nodes[index]["return_array"]
    transport.stop()


class DummyNodeMaxStat():
    def return_arrays(self):
        x = np.zeros((13,), dtype=np.int32)
        x[0] = 0
        y = np.zeros((14,), dtype=np.int32)
        y[0] = 1
        return x, y


@pytest.mark.parametrize("parallel", [False, True])
def test_transport_max_stat_multiple_outputs(parallel):
    transport = Transport([Signature(DummyNodeMaxStat)],
                          parallel=parallel,
                          number_of_processes=1)
    
    outputs = transport.call_nodes_method(node_method="return_arrays")
    assert len(outputs) == 1
    assert outputs[0][0][0] == 0 and outputs[0][1][0] == 1
    
    stat_from_nodes = transport.get_stat_from_nodes()
    max_stat_from_nodes = transport.get_max_stat_from_nodes()
    assert stat_from_nodes[0]["return_arrays"] == 13 * 32 + 14 * 32
    assert max_stat_from_nodes["return_arrays"] == stat_from_nodes[0]["return_arrays"]
    transport.stop()


def test_find_total():
    dct = [{"method_baz": 43}, {"method_foo": 21}]
    assert find_total(dct) == 43 + 21


def test_find_max():
    dct = [{"method_baz": 43}, {"method_foo": 21}]
    assert find_max(dct) == 43
