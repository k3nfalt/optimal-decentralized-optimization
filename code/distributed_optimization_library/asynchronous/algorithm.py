import heapq
import math

import numpy as np

from distributed_optimization_library.factory import Factory
from distributed_optimization_library.function import OptimizationProblemMeta
from distributed_optimization_library.signature import Signature
from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.algorithm import bernoulli_sample


class FactoryAsyncMaster(Factory):
    pass

class FactoryAsyncNode(Factory):
    pass


class StochasticGradientNodeAlgorithm(object):
    def __init__(self, function, **kwargs):
        self._function = function
    
    def cost_calculate_stochastic_gradient(self, point):
        return 1.
    
    def calculate_stochastic_gradient(self, point):
        return self._function.stochastic_gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def cost_calculate_batch_gradient_at_points(self, points, batch_size):
        return len(points) * batch_size
    
    def calculate_batch_gradient_at_points(self, points, batch_size):
        return self._function.batch_gradient_at_points(points, batch_size)
    
    def cost_calculate_batch_gradient_at_points_with_indices(self, points, indices):
        return len(points) * len(indices)
    
    def calculate_batch_gradient_at_points_with_indices(self, points, indices):
        return self._function.batch_gradient_at_points_with_indices(points, indices)
    
    def number_of_functions(self):
        return self._function.number_of_functions()


@FactoryAsyncNode.register("asynchronous_sgd_node")
class AsynchronousSGDNode(StochasticGradientNodeAlgorithm):
    pass


@FactoryAsyncNode.register("rennala_node")
class AsynchronousMiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    pass


@FactoryAsyncNode.register("minibatch_sgd_node")
class MiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    pass


@FactoryAsyncNode.register("rennala_page_node")
class AsynchronousPageMiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    pass


@FactoryAsyncNode.register("minibatch_page_node")
class AsynchronousPageMiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    pass


class BaseMasterAlgorithm(object):
    def __init__(self, homogeneous=False):
        self._homogeneous = homogeneous
    
    def stop(self):
        pass
    
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def calculate_function(self):
        if self._homogeneous:
            return self._transport.call_node_method(node_index=0, node_method='calculate_function',
                                                    point=self._point)
        else:
            return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                             point=self._point))


@FactoryAsyncMaster.register("asynchronous_sgd_master")
class AsynchronousSGD(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None, meta=None):
        super().__init__(homogeneous=True)
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))
    
    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index, "calculate_stochastic_gradient")
        if iter >= self._iter - self._number_of_nodes:
            self._point = self._point - self._gamma * stochastic_gradient
        self._iter += 1
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))


@FactoryAsyncMaster.register("rennala_master")
class AsynchronousMiniBatchSGD(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        super().__init__(homogeneous=True)
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter))
            
        self._gradient_estimator = 0
        self._current_batch = 0
    
    def step(self):
        available_time, node_index, iter = heapq.heappop(self._heap)
        self._time = available_time
        stochastic_gradient = self._transport.call_ready_node(self._time, node_index, "calculate_stochastic_gradient")
        if iter == self._iter:
            self._gradient_estimator = self._gradient_estimator + stochastic_gradient
            self._current_batch += 1
        if self._current_batch >= self._batch_size:
            assert self._current_batch == self._batch_size
            self._point = self._point - self._gamma * (self._gradient_estimator / self._current_batch)
            self._iter += 1
            self._current_batch = 0
            self._gradient_estimator = 0
        available_time = self._transport.call_available_node_method(
            self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
        heapq.heappush(self._heap, (available_time, node_index, self._iter))


def _get_number_of_functions(transport):
    number_of_functions = transport.call_nodes_method(node_method="number_of_functions")
    assert len(number_of_functions) >= 1
    for l, r in zip(number_of_functions[:-1], number_of_functions[1:]):
        assert l == r
    return number_of_functions[0]


@FactoryAsyncMaster.register("rennala_page_master")
class AsynchronousPageMiniBatchSGD(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, inner_batch_size_full=1, batch_size=None, seed=None, meta=None):
        super().__init__(homogeneous=True)
        self._transport = transport
        self._point = point
        self._previous_point = None
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._inner_batch_size = 1
        self._inner_batch_size_full = inner_batch_size_full
        self._seed = seed
        self._time = 0
        self._generator = np.random.default_rng(seed)
        self._number_of_functions = _get_number_of_functions(self._transport)
        self._batch_size = math.ceil(math.sqrt(self._number_of_functions)) if self._batch_size is None else self._batch_size
        print(f"# of functions {self._number_of_functions}, batch size {self._batch_size}")
        self._prob = self._batch_size / (self._batch_size + self._number_of_functions)
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._gradient_estimator = self._calculate_full_gradient()
    
    def step(self):
        self._previous_point = self._point
        self._point = self._point - self._gamma * self._gradient_estimator
        calculate_checkpoint = bernoulli_sample(self._generator, self._prob)
        if calculate_checkpoint:
            self._gradient_estimator = self._calculate_full_gradient()
        else:
            heap = []
            number_of_received_gradients = 0
            aggregated_gradient = 0
            for node_index in range(self._transport.get_number_of_nodes()):
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_batch_gradient_at_points", force_lazy=True,
                    points=(self._point, self._previous_point), batch_size=self._inner_batch_size)
                heapq.heappush(heap, (available_time, node_index))
            while number_of_received_gradients < self._batch_size:
                available_time, node_index = heapq.heappop(heap)
                self._time = available_time
                gradient, previous_gradient = self._transport.call_ready_node(self._time, node_index, "calculate_batch_gradient_at_points")
                aggregated_gradient = aggregated_gradient + self._inner_batch_size * (gradient - previous_gradient)
                number_of_received_gradients += self._inner_batch_size
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_batch_gradient_at_points", force_lazy=True,
                    points=(self._point, self._previous_point), batch_size=self._inner_batch_size)
                heapq.heappush(heap, (available_time, node_index))
            aggregated_gradient = aggregated_gradient / number_of_received_gradients
            self._gradient_estimator = self._gradient_estimator + aggregated_gradient
            self._transport.stop_all_calculations()
        
    def _calculate_full_gradient(self):
        assert self._number_of_functions % self._inner_batch_size_full == 0
        required_groups = np.split(range(self._number_of_functions), 
                                   np.arange(self._inner_batch_size_full,
                                             self._number_of_functions,
                                             self._inner_batch_size_full))
        heap = []
        for node_index in range(self._transport.get_number_of_nodes()):
            group_index = self._generator.integers(len(required_groups), size=1)[0]
            indices = required_groups[group_index]
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_batch_gradient_at_points_with_indices", force_lazy=True,
                points=[self._point], indices=indices)
            heapq.heappush(heap, (available_time, node_index, group_index, len(indices)))
        number_of_received_gradients = 0
        aggregated_gradient = 0
        
        id_to_required_indices = dict({v: v for v in range(len(required_groups))})
        required_indices_to_id = dict({v: v for v in range(len(required_groups))})
        
        while len(required_indices_to_id) > 0:
            available_time, node_index, group_index, batch_size = heapq.heappop(heap)
            self._time = available_time
            gradient = self._transport.call_ready_node(self._time, node_index, "calculate_batch_gradient_at_points_with_indices")[0]
            if group_index in required_indices_to_id:
                aggregated_gradient = aggregated_gradient + batch_size * gradient
                number_of_received_gradients += batch_size
                
                empty_id = required_indices_to_id.pop(group_index)
                largest_id = len(required_indices_to_id)
                if empty_id != largest_id:
                    largest_key = id_to_required_indices[largest_id]
                    required_indices_to_id[largest_key] = empty_id
                    id_to_required_indices[empty_id] = largest_key
                id_to_required_indices.pop(largest_id)
            
            if len(required_indices_to_id) == 0:
                break
            random_index = self._generator.integers(len(required_indices_to_id), size=1)[0]
            group_index = id_to_required_indices[random_index]
            indices = required_groups[group_index]
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_batch_gradient_at_points_with_indices", force_lazy=True,
                points=[self._point], indices=indices)
            heapq.heappush(heap, (available_time, node_index, group_index, len(indices)))
        assert number_of_received_gradients == self._number_of_functions
        aggregated_gradient = aggregated_gradient / number_of_received_gradients
        self._transport.stop_all_calculations()
        return aggregated_gradient


@FactoryAsyncMaster.register("minibatch_sgd_master")
class MiniBatchSGD(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        super().__init__(homogeneous=True)
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._current_times = [None for _ in range(self._number_of_nodes)]
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            self._current_times[node_index] = available_time
    
    def step(self):
        max_available_time = -np.inf
        for node_index in range(self._number_of_nodes):
            available_time = self._current_times[node_index]
            max_available_time = max(max_available_time, available_time)
            
        self._time = max_available_time
        gradient_estimator = 0
        for node_index in range(self._number_of_nodes):
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index, "calculate_stochastic_gradient")
            gradient_estimator = gradient_estimator + stochastic_gradient
        self._point = self._point - self._gamma * (gradient_estimator / self._number_of_nodes)
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_stochastic_gradient", point=self._point)
            self._current_times[node_index] = available_time


@FactoryAsyncMaster.register("minibatch_page_master")
class MiniBatchPAGE(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, batch_size=None, seed=None, meta=None):
        super().__init__(homogeneous=True)
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._batch_size = batch_size
        self._seed = seed
        self._time = 0
        self._generator = np.random.default_rng(seed)
        self._number_of_functions = _get_number_of_functions(self._transport)
        self._batch_size = math.ceil(math.sqrt(self._number_of_functions)) if self._batch_size is None else self._batch_size
        print(f"# of functions {self._number_of_functions}, batch size {self._batch_size}")
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._prepare_required_groups()
        self._prepare_batch_size_per_worker()
        self._gradient_estimator = self._calculate_full_gradient()
        self._prob = self._batch_size / (self._batch_size + self._number_of_functions)
    
    def step(self):
        self._previous_point = self._point
        self._point = self._point - self._gamma * self._gradient_estimator
        calculate_checkpoint = bernoulli_sample(self._generator, self._prob)
        if calculate_checkpoint:
            self._gradient_estimator = self._calculate_full_gradient()
        else:
            self._generator.shuffle(self._batch_size_per_worker)
            current_times = [None for _ in range(self._number_of_nodes)]
            for node_index in range(self._number_of_nodes):
                if self._batch_size_per_worker[node_index] == 0:
                    continue
                available_time = self._transport.call_available_node_method(
                    self._time, node_index, node_method="calculate_batch_gradient_at_points", force_lazy=True,
                    points=(self._point, self._previous_point), 
                    batch_size=self._batch_size_per_worker[node_index])
                current_times[node_index] = available_time
            self._time = np.max(list(filter(lambda x: x is not None, current_times)))
            
            number_of_received_gradients = 0
            aggregated_gradient = 0
            for node_index in range(self._number_of_nodes):
                if self._batch_size_per_worker[node_index] == 0:
                    continue
                gradient, previous_gradient = self._transport.call_ready_node(
                    self._time, node_index, "calculate_batch_gradient_at_points")
                aggregated_gradient = aggregated_gradient + self._batch_size_per_worker[node_index] * (gradient - previous_gradient)
                number_of_received_gradients += self._batch_size_per_worker[node_index]
            assert number_of_received_gradients == self._batch_size
            aggregated_gradient /= number_of_received_gradients
            self._gradient_estimator = self._gradient_estimator + aggregated_gradient
        
    def _calculate_full_gradient(self):
        self._generator.shuffle(self._required_groups)
        current_times = [None] * self._transport.get_number_of_nodes()
        batch_sizes = [None] * self._transport.get_number_of_nodes()
        for node_index in range(self._transport.get_number_of_nodes()):
            if len(self._required_groups[node_index]) == 0:
                continue
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method="calculate_batch_gradient_at_points_with_indices", force_lazy=True,
                points=[self._point], indices=self._required_groups[node_index])
            current_times[node_index] = available_time
            batch_sizes[node_index] = len(self._required_groups[node_index])
        
        number_of_received_gradients = 0
        aggregated_gradient = 0
        self._time = np.max(list(filter(lambda x: x is not None, current_times)))
        for node_index in range(self._transport.get_number_of_nodes()):
            if len(self._required_groups[node_index]) == 0:
                continue
            gradient = self._transport.call_ready_node(self._time, node_index, "calculate_batch_gradient_at_points_with_indices")[0]
            aggregated_gradient = aggregated_gradient + batch_sizes[node_index] * gradient
            number_of_received_gradients += batch_sizes[node_index]
        assert number_of_received_gradients == self._number_of_functions
        aggregated_gradient = aggregated_gradient / number_of_received_gradients
        return aggregated_gradient

    def _prepare_required_groups(self):
        self._required_groups = [[] for _ in range(self._number_of_nodes)]
        for index in range(self._number_of_functions):
            self._required_groups[index % self._number_of_nodes].append(index)
            
    def _prepare_batch_size_per_worker(self):
        self._batch_size_per_worker = [0 for _ in range(self._number_of_nodes)]
        for index in range(self._batch_size):
            self._batch_size_per_worker[index % self._number_of_nodes] += 1


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
    nodes = [Signature(node_cls, function, seed=_generate_seed(generator), **algorithm_node_params) 
             for function in functions]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    return master_cls(transport, point, seed=seed, meta=meta, **algorithm_master_params)
