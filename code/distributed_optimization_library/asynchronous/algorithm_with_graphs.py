import numpy as np
import scipy.sparse
import heapq
from collections import defaultdict
from enum import Enum
from copy import copy

from distributed_optimization_library.factory import Factory
from distributed_optimization_library.function import OptimizationProblemMeta
from distributed_optimization_library.signature import Signature
from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport


def add_torus(delays, num_nodes, distance=1):
    num_nodes_on_edge = int(np.sqrt(num_nodes))
    np.testing.assert_almost_equal(num_nodes_on_edge ** 2, num_nodes)
    def _get_id(i, j):
        return i + j * num_nodes_on_edge
    for i in range(num_nodes_on_edge):
        for j in range(num_nodes_on_edge):
            to_node = _get_id(i, j)
            delays[f"send_vector_to_{to_node}"] = [np.inf] * num_nodes
            delays[f"send_vector_to_{to_node}"][to_node] = 0
            if j - 1 >= 0:
                up_node = _get_id(i, j - 1)
                delays[f"send_vector_to_{to_node}"][up_node] = distance
            if j + 1 < num_nodes_on_edge:
                down_node = _get_id(i, j + 1)
                delays[f"send_vector_to_{to_node}"][down_node] = distance
            if i - 1 >= 0:
                left_node = _get_id(i - 1, j)
                delays[f"send_vector_to_{to_node}"][left_node] = distance
            if i + 1 < num_nodes_on_edge:
                right_node = _get_id(i + 1, j)
                delays[f"send_vector_to_{to_node}"][right_node] = distance


class FactoryAsyncMaster(Factory):
    pass

class FactoryAsyncNode(Factory):
    pass


class Vector(object):
    def __init__(self, vector, id=-1):
        self._vector = vector
        self._num_add = 1
        self._ids = defaultdict(int)
        self._ids[id] = 1

    def __add__(self, vector):
        sum_vector = Vector(self._vector + vector._vector)
        sum_vector._num_add = self._num_add + vector._num_add
        sum_vector._ids = copy(self._ids)
        for k in vector._ids.keys():
            sum_vector._ids[k] = sum_vector._ids[k] + vector._ids[k]
        return sum_vector

    def __len__(self):
        return len(self._vector)


class StochasticGradientNodeAlgorithm(object):
    def __init__(self, function, num_nodes, id=-1, **kwargs):
        self._function = function
        self.clear()
        self._id = id
        for to_node in range(num_nodes):
            setattr(self, f"cost_send_vector_to_{to_node}", self._cost_send_vector)
            setattr(self, f"send_vector_to_{to_node}", self._send_vector)
    
    def cost_calculate_stochastic_gradient(self):
        return 1.
    
    def calculate_stochastic_gradient(self):
        self._calculated_stochastic_gradient =\
            Vector(self._function.stochastic_gradient(self._point), id=self._id)
        
    def _cost_send_vector(self, broadcast=False, **kwargs):
        if not broadcast:
            return len(self._vector_to_send)
        else:
            return len(self._vector_to_broadcast)
        
    def _send_vector(self, broadcast=False, reset=True):
        if not broadcast:
            assert self._vector_to_send is not None
            vector_to_send = self._vector_to_send
            if reset:
                self._vector_to_send = None
        else:
            assert self._vector_to_broadcast is not None
            vector_to_send = self._vector_to_broadcast
            if reset:
                self._vector_to_broadcast = None
        return vector_to_send
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
        
    def receive_vector(self, vector):
        raise NotImplementedError()
    
    def send_vector_to_pivot_worker(self, vector):
        self._vector_to_broadcast = vector
        
    def clear(self):
        self._calculated_stochastic_gradient = None
        self._vector_to_send = None
        self._vector_to_broadcast = None
        self._point = None
    
    def number_of_stochastic_gradients(self):
        return self._vector_to_send._num_add if self._vector_to_send is not None else 0


class BaseMasterAlgorithm(object):
    WEIRD_CONST = -9999
    def __init__(self, transport):
        self._transport = transport
    
    def stop(self):
        pass
    
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))

    def send_vector(self, from_node, to_node, broadcast=False, reset=True):
        available_time = self._transport.call_available_node_method(
            self._time, from_node, 
            node_method=f"send_vector_to_{to_node}",
            broadcast=broadcast,
            reset=reset)
        return available_time

    def receive_vector(self, from_node, to_node, broadcast=False):
        vector = self._transport.call_ready_node(
            self._time, from_node, node_method=f"send_vector_to_{to_node}")
        self._transport.call_node_method(to_node, "receive_vector", vector=vector,
                                         broadcast=broadcast)
        
    def receive_vector_from_pivot_worker(self):
        available_time = self._transport.call_available_node_method(
            self._time, self._pivot_worker, 
            node_method=f"send_vector_to_{self._pivot_worker}")
        assert available_time == self._time
        vector = self._transport.call_ready_node(
            self._time, self._pivot_worker, 
            node_method=f"send_vector_to_{self._pivot_worker}")
        return vector
    
    def send_vector_to_pivot_worker(self, vector):
        self._transport.call_node_method(self._pivot_worker, 
                                         "send_vector_to_pivot_worker", 
                                         vector=vector)
    
    # todo: test it
    def _parse_graph(self):
        delays = self._transport.get_delays()
        methods = list(delays.keys())
        methods = filter(lambda m: "send_vector_to_" in m, methods)
        num_nodes = self._transport.get_number_of_nodes()
        rho = scipy.sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
        for method in methods:
            substr = method.split('_')
            to_node = int(substr[3])
            for from_node in range(len(delays[method])):
                rho[from_node, to_node] = delays[method][from_node]
        for row, col in zip(*rho.nonzero()):
            assert rho[row, col] == rho[col, row]
        dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(
            rho, directed=False, return_predecessors=True)
        max_dist_matrix = np.max(dist_matrix, axis=1)
        self._pivot_worker = np.argmin(max_dist_matrix)
        self._distance = dist_matrix[self._pivot_worker, :]
        self._predecessors = predecessors[self._pivot_worker, :]
        
    def _prepare_inverse_predecessors(self):
        num_nodes = self._transport.get_number_of_nodes()
        visited = [False] * num_nodes
        visited[self._pivot_worker] = True
        self._inverse_predecessors = [[] for _ in range(num_nodes)]
        def add_to_next(node_index):
            if not visited[self.next(node_index)]:
                add_to_next(self.next(node_index))
            self._inverse_predecessors[self.next(node_index)].append(node_index)
            visited[node_index] = True
        for node_index in range(num_nodes):
            if not visited[node_index] and node_index != self._pivot_worker:
                add_to_next(node_index)
        
    def next(self, node):
        return self._predecessors[node]


@FactoryAsyncNode.register("minibatch_sgd_node")
class MiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    def receive_vector(self, vector, broadcast=False):
        if not broadcast:
            if self._vector_to_send is None:
                self._vector_to_send = vector
            else:
                self._vector_to_send = self._vector_to_send + vector
        else:
            self._vector_to_broadcast = vector
    
    def move_gradient_to_buffer(self):
        if self._vector_to_send is None:
            self._vector_to_send = self._calculated_stochastic_gradient
        else:
            self._vector_to_send = self._vector_to_send + self._calculated_stochastic_gradient
    
    def move_buffer_to_point(self):
        assert self._vector_to_broadcast is not None
        self._point = np.copy(self._vector_to_broadcast)


@FactoryAsyncNode.register("fragile_node")
class FragileSGDNode(MiniBatchSGDNode):
    pass
    

@FactoryAsyncMaster.register("minibatch_sgd_master")
class MiniBatchSGD(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None, seed=None):
        super().__init__(transport)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._parse_graph()
        self._prepare_inverse_predecessors()
    
    def step(self):
        self._transport.call_nodes_method("clear")
        self._transport.stop_all_calculations()
        self.send_vector_to_pivot_worker(self._point)
        self._transport.call_node_method(self._pivot_worker, "move_buffer_to_point")
        heap = []
        for node_index in self._inverse_predecessors[self._pivot_worker]:
            available_time = self.send_vector(self._pivot_worker, node_index, 
                                              broadcast=True, reset=False)
            heapq.heappush(heap, (available_time, self._pivot_worker, node_index))
        while len(heap) > 0:
            available_time, from_node, to_node = heapq.heappop(heap)
            self._time = available_time
            self.receive_vector(from_node, to_node, broadcast=True)
            self._transport.call_node_method(to_node, "move_buffer_to_point")
            for node_index in self._inverse_predecessors[to_node]:
                available_time = self.send_vector(to_node, node_index, 
                                                  broadcast=True, reset=False)
                heapq.heappush(heap, (available_time, to_node, node_index))
        available_times = [None for _ in range(self._number_of_nodes)]
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient")
            available_times[node_index] = available_time
        max_available_time = np.max(available_times)
        self._time = max_available_time
        for node_index in range(self._number_of_nodes):
            self._transport.call_ready_node(self._time, node_index, "calculate_stochastic_gradient")
            self._transport.call_node_method(node_index, "move_gradient_to_buffer")
        class Task(Enum):
            RECEIVE = 1
            SEND = 2
        heap = []
        working_nodes = [None] * self._number_of_nodes
        waiting_tasks = set()
        for node_index in range(self._number_of_nodes):
            heapq.heappush(heap, (self._time, node_index, Task.SEND.value))
            waiting_tasks.add(node_index)
        while len(heap) > 0:
            available_time, node_index, task = heapq.heappop(heap)
            self._time = available_time
            if task == Task.SEND.value and self.next(node_index) != self.WEIRD_CONST:
                available_time = self.send_vector(node_index, self.next(node_index))
                working_nodes[node_index] = available_time
                assert node_index in waiting_tasks
                waiting_tasks.remove(node_index)
                heapq.heappush(heap, (available_time, node_index, Task.RECEIVE.value))
            if task == Task.RECEIVE.value:
                self.receive_vector(node_index, self.next(node_index))
                if working_nodes[self.next(node_index)] is not None:
                    time_send = working_nodes[self.next(node_index)]
                else:
                    time_send = self._time
                if self.next(node_index) not in waiting_tasks:
                    heapq.heappush(heap, (time_send, self.next(node_index), Task.SEND.value))
                    waiting_tasks.add(self.next(node_index))
        gradient = self.receive_vector_from_pivot_worker()
        assert gradient._num_add == self._number_of_nodes
        self._point = self._point - self._gamma * gradient._vector / gradient._num_add


@FactoryAsyncMaster.register("fragile_master")
class FragileSGD(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, batch_size=None, meta=None, seed=None):
        super().__init__(transport)
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        self._batch_size = batch_size
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._parse_graph()
        self._prepare_inverse_predecessors()
    
    def step(self):
        class Task(Enum):
            RECEIVE = 1
            SEND = 2
            BROADCAST = 3
            RECEIVE_BROADCAST = 4
            FINISH_CALCULATION = 5
            PREPARE_SEND = 6
            
        self._transport.call_nodes_method("clear")
        self._transport.stop_all_calculations()
        
        self.send_vector_to_pivot_worker(self._point)
        self._transport.call_node_method(self._pivot_worker, "move_buffer_to_point")
        available_time = self._transport.call_available_node_method(
            self._time, self._pivot_worker, node_method="calculate_stochastic_gradient")
        heap = []
        heapq.heappush(heap, (available_time, Task.FINISH_CALCULATION.value, self._pivot_worker))
        for node_index in self._inverse_predecessors[self._pivot_worker]:
            available_time = self.send_vector(self._pivot_worker, node_index, 
                                              broadcast=True, reset=False)
            heapq.heappush(heap, (available_time, Task.RECEIVE_BROADCAST.value, (self._pivot_worker, node_index)))
        
        working_nodes = [None] * self._number_of_nodes
        nodes_empty = [True] * self._number_of_nodes
        waiting_tasks = set()
        number_of_stochastic_gradients = 0
        while number_of_stochastic_gradients < self._batch_size:
            available_time, task, input_task = heapq.heappop(heap)
            self._time = available_time
            if task == Task.RECEIVE_BROADCAST.value:
                from_node, to_node = input_task
                self.receive_vector(from_node, to_node, broadcast=True)
                self._transport.call_node_method(to_node, "move_buffer_to_point")
                available_time = self._transport.call_available_node_method(
                    self._time, to_node, node_method="calculate_stochastic_gradient")
                heapq.heappush(heap, (available_time, Task.FINISH_CALCULATION.value, to_node))
                for node_index in self._inverse_predecessors[to_node]:
                    available_time = self.send_vector(to_node, node_index, 
                                                      broadcast=True, reset=False)
                    heapq.heappush(heap, (available_time, Task.RECEIVE_BROADCAST.value, (to_node, node_index)))
            if task == Task.FINISH_CALCULATION.value:
                node_index = input_task
                self._transport.call_ready_node(self._time, node_index, "calculate_stochastic_gradient")
                nodes_empty[node_index] = False
                self._transport.call_node_method(node_index, "move_gradient_to_buffer")
                if self.next(node_index) != self.WEIRD_CONST:
                    heapq.heappush(heap, (self._time, Task.PREPARE_SEND.value, node_index))
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_stochastic_gradient")
                heapq.heappush(heap, (available_time, Task.FINISH_CALCULATION.value, node_index))
            if task == Task.SEND.value:
                node_index = input_task
                if self.next(node_index) == self.WEIRD_CONST:
                    continue
                available_time = self.send_vector(node_index, self.next(node_index))
                working_nodes[node_index] = available_time
                nodes_empty[node_index] = True
                assert node_index in waiting_tasks
                waiting_tasks.remove(node_index)
                heapq.heappush(heap, (available_time, Task.RECEIVE.value, node_index))
            if task == Task.RECEIVE.value:
                node_index = input_task
                self.receive_vector(node_index, self.next(node_index))
                nodes_empty[node_index] = False
                working_nodes[node_index] = None
                heapq.heappush(heap, (self._time, Task.PREPARE_SEND.value, self.next(node_index)))
            if task == Task.PREPARE_SEND.value:
                node_index = input_task
                if working_nodes[node_index] is not None:
                    time_send = working_nodes[node_index]
                else:
                    time_send = self._time
                if node_index not in waiting_tasks and not nodes_empty[node_index]:
                    heapq.heappush(heap, (time_send, Task.SEND.value, node_index))
                    waiting_tasks.add(node_index)
            number_of_stochastic_gradients =\
                self._transport.call_node_method(self._pivot_worker, "number_of_stochastic_gradients")
        gradient = self.receive_vector_from_pivot_worker()
        assert gradient._num_add >= self._batch_size
        self._point = self._point - self._gamma * gradient._vector / gradient._num_add


def _generate_seed(generator):
    return generator.integers(10e9)


def get_algorithm(functions, point, seed, 
                  algorithm_name, delays, 
                  algorithm_master_params={}, algorithm_node_params={},
                  meta=OptimizationProblemMeta()):
    node_name = algorithm_name + "_node"
    master_name = algorithm_name + "_master"
    node_cls = FactoryAsyncNode.get(node_name)
    master_cls = FactoryAsyncMaster.get(master_name)
    generator = np.random.default_rng(seed)
    nodes = [Signature(node_cls, function, len(functions), id=index, 
                       seed=_generate_seed(generator), **algorithm_node_params)
             for index, function in enumerate(functions)]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    return master_cls(transport, point, seed=seed, meta=meta, **algorithm_master_params)
