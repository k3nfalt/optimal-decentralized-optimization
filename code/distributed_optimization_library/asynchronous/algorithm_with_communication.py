import heapq

import numpy as np
import scipy.optimize

from distributed_optimization_library.factory import Factory
from distributed_optimization_library.function import OptimizationProblemMeta
from distributed_optimization_library.signature import Signature
from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.compressor import RandKCompressor, CompressedVector


class FactoryAsyncMaster(Factory):
    pass

class FactoryAsyncNode(Factory):
    pass


def get_compressor(number_of_coordinates, seed, dim):
    return RandKCompressor(number_of_coordinates, seed, dim)


def get_omega(number_of_coordinates, dim):
    seed = 42
    comp = get_compressor(number_of_coordinates, seed, dim)
    return comp.omega()


class StochasticGradientNodeAlgorithm(object):
    def __init__(self, function, **kwargs):
        self._function = function
        self._calculated_stochastic_gradient = None
        self._vector_to_send = None
    
    def cost_calculate_stochastic_gradient(self, *args, **kwargs):
        return 1.
    
    def calculate_stochastic_gradient(self, point):
        self._calculated_stochastic_gradient = self._function.stochastic_gradient(point)
        
    def cost_send_vector(self, *args, **kwargs):
        if isinstance(self._vector_to_send, np.ndarray):
            return len(self._vector_to_send)
        if isinstance(self._vector_to_send, CompressedVector):
            return self._vector_to_send.number_of_elements()
        else:
            assert False
        
    def send_vector(self):
        assert self._vector_to_send is not None
        vector_to_send = self._vector_to_send
        self._vector_to_send = None
        return vector_to_send
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


METHOD_GRADIENT = "calculate_stochastic_gradient"
METHOD_SEND = "send_vector"


class BaseMasterAlgorithm(object):
    def stop(self):
        pass
    
    def get_point(self):
        return self._point
    
    def get_time(self):
        return self._time
    
    def calculate_function(self):
        return np.mean(self._transport.call_nodes_method(node_method='calculate_function',
                                                         point=self._point))


@FactoryAsyncNode.register("minibatch_sgd_node")
class MiniBatchSGDNode(StochasticGradientNodeAlgorithm):
    def move_gradient_to_buffer(self):
        self._vector_to_send = self._calculated_stochastic_gradient


@FactoryAsyncMaster.register("minibatch_sgd_master")
class MiniBatchSGD(BaseMasterAlgorithm):
    _MOVE_VECTOR = "move_gradient_to_buffer"
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None, seed=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._methods = self._transport.get_methods()
        self._current_times = {k: [None] * self._number_of_nodes for k in self._methods}
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            self._current_times[METHOD_GRADIENT][node_index] = available_time
    
    def step(self):
        indices_sorted = np.argsort(self._current_times[METHOD_GRADIENT])
        for node_index in indices_sorted:
            self._time = self._current_times[METHOD_GRADIENT][node_index]
            self._transport.call_ready_node(self._time, node_index, METHOD_GRADIENT)
            self._transport.call_node_method(node_index, self._MOVE_VECTOR)
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_SEND)
            self._current_times[METHOD_SEND][node_index] = available_time
        gradient_estimator = 0
        indices_sorted = np.argsort(self._current_times[METHOD_SEND])
        for node_index in indices_sorted:
            self._time = max(self._time, self._current_times[METHOD_SEND][node_index])
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index, METHOD_SEND)
            gradient_estimator = gradient_estimator + stochastic_gradient
        self._point = self._point - self._gamma * (gradient_estimator / self._number_of_nodes)
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            self._current_times[METHOD_GRADIENT][node_index] = available_time


@FactoryAsyncNode.register("fastest_sgd_node")
class FastestSGDNode(StochasticGradientNodeAlgorithm):
    def send_vector_locally(self):
        return self._calculated_stochastic_gradient


@FactoryAsyncMaster.register("fastest_sgd_master")
class FastestSGD(BaseMasterAlgorithm):
    _GET_VECTOR = "send_vector_locally"
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None, seed=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        
        delays = self._transport.get_delays()
        self._fastest_node = np.argmin(delays[METHOD_GRADIENT])
        self._current_time = self._transport.call_available_node_method(
            self._time, self._fastest_node, node_method=METHOD_GRADIENT, point=self._point)
    
    def step(self):
        self._time = self._current_time
        self._transport.call_ready_node(self._time, self._fastest_node, METHOD_GRADIENT)
        stochastic_gradient = self._transport.call_node_method(self._fastest_node, self._GET_VECTOR)
        self._point = self._point - self._gamma * stochastic_gradient
        self._current_time = self._transport.call_available_node_method(
            self._time, self._fastest_node, node_method=METHOD_GRADIENT, point=self._point)


@FactoryAsyncNode.register("qsgd_node")
class QSGDNode(StochasticGradientNodeAlgorithm):
    def init_compressor(self, number_of_coordinates, seed, dim):
        self._compressor = get_compressor(number_of_coordinates, seed, dim)
    
    def move_gradient_to_buffer_and_compress(self):
        self._vector_to_send = self._compressor.compress(self._calculated_stochastic_gradient)


@FactoryAsyncMaster.register("qsgd_master")
class QSGD(BaseMasterAlgorithm):
    _MOVE_VECTOR = "move_gradient_to_buffer_and_compress"
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None, seed=None,
                 number_of_coordinates=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        self._seed = np.random.default_rng(seed)
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._methods = self._transport.get_methods()
        self._current_times = {k: [None] * self._number_of_nodes for k in self._methods}
        for node_index in range(self._number_of_nodes):
            self._transport.call_node_method(node_index, "init_compressor",
                                             number_of_coordinates=number_of_coordinates, 
                                             seed=self._seed, dim=len(self._point))
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            self._current_times[METHOD_GRADIENT][node_index] = available_time
    
    def step(self):
        indices_sorted = np.argsort(self._current_times[METHOD_GRADIENT])
        for node_index in indices_sorted:
            self._time = self._current_times[METHOD_GRADIENT][node_index]
            self._transport.call_ready_node(self._time, node_index, METHOD_GRADIENT)
            self._transport.call_node_method(node_index, self._MOVE_VECTOR)
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_SEND)
            self._current_times[METHOD_SEND][node_index] = available_time
        gradient_estimator = 0
        indices_sorted = np.argsort(self._current_times[METHOD_SEND])
        for node_index in indices_sorted:
            self._time = max(self._time, self._current_times[METHOD_SEND][node_index])
            compressed_gradient = self._transport.call_ready_node(self._time, node_index, METHOD_SEND)
            stochastic_gradient = compressed_gradient.decompress()
            gradient_estimator = gradient_estimator + stochastic_gradient
        self._point = self._point - self._gamma * (gradient_estimator / self._number_of_nodes)
        for node_index in range(self._number_of_nodes):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            self._current_times[METHOD_GRADIENT][node_index] = available_time


@FactoryAsyncNode.register("asynchronous_sgd_node")
class AsynchronousSGDNode(StochasticGradientNodeAlgorithm):
    def move_gradient_to_buffer(self):
        self._vector_to_send = self._calculated_stochastic_gradient


@FactoryAsyncMaster.register("asynchronous_sgd_master")
class AsynchronousSGD(BaseMasterAlgorithm):
    _MOVE_VECTOR = "move_gradient_to_buffer"
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None, meta=None):
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        
        self._heap = []
        self._iter = 0
        self._number_of_nodes = self._transport.get_number_of_nodes()
        
        for node_index in range(self._transport.get_number_of_nodes()):
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter, METHOD_GRADIENT))
    
    def step(self):
        available_time, node_index, iter, status = heapq.heappop(self._heap)
        self._time = available_time
        if status == METHOD_SEND:
            stochastic_gradient = self._transport.call_ready_node(self._time, node_index, METHOD_SEND)
            if iter >= self._iter - self._number_of_nodes:
                self._point = self._point - self._gamma * stochastic_gradient
            self._iter += 1
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            heapq.heappush(self._heap, (available_time, node_index, self._iter, METHOD_GRADIENT))
        else:
            assert status == METHOD_GRADIENT
            self._transport.call_ready_node(self._time, node_index, METHOD_GRADIENT)
            self._transport.call_node_method(node_index, self._MOVE_VECTOR)
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_SEND)
            heapq.heappush(self._heap, (available_time, node_index, iter, METHOD_SEND))


def find_equilibrium_time(computation_times, communication_times_per_coor, 
                          number_of_coordinates, dim, sigma_eps):
    _EPS = 1e-8
    computation_times = np.array(computation_times)
    communication_times_per_coor = np.array(communication_times_per_coor)
    communication_times = number_of_coordinates * communication_times_per_coor
    omega = get_omega(number_of_coordinates, dim)
    max_times = np.maximum(computation_times, communication_times)
    index_sort = np.argsort(max_times)
    sort_max = max_times[index_sort]
    computation_times = computation_times[index_sort]
    communication_times = communication_times[index_sort]
    num_nodes = len(communication_times)
    budgets = [None] * num_nodes
    for j in range(num_nodes):
        cop = computation_times[:j + 1]
        com = communication_times[:j + 1]
        
        den_init = (2 * com * omega + 2 * cop * sigma_eps)
        if np.any(np.abs(den_init) <= _EPS):
            budgets[j] = sort_max[j]
            continue
        init_point = 1. / np.sum(1. / den_init)
        
        def _func(s):
            den = (2 * com * omega +
                   4 * com * omega * cop * sigma_eps / s +
                   2 * cop * sigma_eps)
            return  1. / np.sum(1. / den) - s
        budgets[j] = np.maximum(scipy.optimize.fsolve(_func, init_point), sort_max[j])
    return np.min(budgets)


@FactoryAsyncNode.register("shadowheart_sgd_node")
class ShadowheartNode(StochasticGradientNodeAlgorithm):
    def __init__(self, function, **kwargs):
        super().__init__(function, **kwargs)
        self._aggregated_vector = 0
    
    def init_compressor(self, number_of_coordinates, seed, dim):
        self._compressor = get_compressor(number_of_coordinates, seed, dim)
    
    def move_gradient_to_buffer_and_compress(self):
        self._vector_to_send = self._compressor.compress(self._aggregated_vector)
        
    def aggregate(self):
        self._aggregated_vector += self._calculated_stochastic_gradient
        
    def clear_aggregated_vector(self):
        self._aggregated_vector = 0


@FactoryAsyncMaster.register("shadowheart_sgd_master")
class Shadowheart(BaseMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, meta=None, seed=None,
                 number_of_coordinates=None, sigma_eps=None, clip_number_of_communications=False):
        assert sigma_eps is not None
        omega = get_omega(number_of_coordinates, len(point))
        self._transport = transport
        self._point = point
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._time = 0
        self._seed = np.random.default_rng(seed)
        self._heap = []
        self._dim = len(self._point)
        
        self._number_of_nodes = self._transport.get_number_of_nodes()
        delays = self._transport.get_delays()
        computation_times = np.array(delays[METHOD_GRADIENT])
        communication_times_per_coor = np.array(delays[METHOD_SEND])
        assert self._number_of_nodes == len(computation_times)
        assert self._number_of_nodes == len(communication_times_per_coor)
        communication_times = communication_times_per_coor * number_of_coordinates
        self._equilibrium_time = find_equilibrium_time(computation_times, communication_times_per_coor,
                                                       number_of_coordinates, self._dim, sigma_eps)
        self._batch_sizes = np.floor(self._equilibrium_time / computation_times)
        self._number_of_communications = np.floor(self._equilibrium_time / communication_times)
        if clip_number_of_communications:
            self._number_of_communications = np.minimum(self._number_of_communications, 
                                                        np.ceil(self._dim / number_of_coordinates))
        self._active_worker = np.logical_and(self._batch_sizes > 0, self._number_of_communications > 0)
        for node_index in range(self._number_of_nodes):
            self._transport.call_node_method(node_index, "init_compressor",
                                             number_of_coordinates=number_of_coordinates, 
                                             seed=self._seed, dim=self._dim)
        if sigma_eps == 0 and omega == 0:
            self._weights = np.ones(self._number_of_nodes)
        else:
            self._weights = 1 / (self._batch_sizes * omega + 
                                 omega * sigma_eps + 
                                 self._number_of_communications * sigma_eps)
        self._weights[~self._active_worker] = 0
        if False:
            print(self._weights)
            print(self._number_of_communications)
            print(self._batch_sizes)
            assert False
        self._normalize_factor = np.sum(self._weights * self._number_of_communications * self._batch_sizes)
    
    def step(self):
        batch_sizes = np.copy(self._batch_sizes)
        number_of_communications = np.copy(self._number_of_communications)
        for node_index in range(self._number_of_nodes):
            self._transport.call_node_method(node_index, "clear_aggregated_vector")
            if not self._active_worker[node_index]:
                continue
            available_time = self._transport.call_available_node_method(
                self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
            heapq.heappush(self._heap, (available_time, node_index))
        gradient_estimator = 0
        while len(self._heap) > 0:
            self._time, node_index = heapq.heappop(self._heap)
            if batch_sizes[node_index] > 0:
                self._transport.call_ready_node(
                    self._time, node_index, node_method=METHOD_GRADIENT)
                self._transport.call_node_method(node_index, "aggregate")
                batch_sizes[node_index] -= 1
                if batch_sizes[node_index] > 0:
                    available_time = self._transport.call_available_node_method(
                        self._time, node_index, node_method=METHOD_GRADIENT, point=self._point)
                else:
                    self._transport.call_node_method(node_index, "move_gradient_to_buffer_and_compress")
                    available_time = self._transport.call_available_node_method(
                        self._time, node_index, node_method=METHOD_SEND)
                heapq.heappush(self._heap, (available_time, node_index))
            elif number_of_communications[node_index] > 0:
                compressed_vector = self._transport.call_ready_node(
                    self._time, node_index, node_method=METHOD_SEND)
                gradient_estimator += self._weights[node_index] * compressed_vector.decompress()
                number_of_communications[node_index] -= 1
                if number_of_communications[node_index] > 0:
                    self._transport.call_node_method(node_index, "move_gradient_to_buffer_and_compress")
                    available_time = self._transport.call_available_node_method(
                        self._time, node_index, node_method=METHOD_SEND)
                    heapq.heappush(self._heap, (available_time, node_index))
        self._point = self._point - self._gamma * (gradient_estimator / self._normalize_factor)


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
