import contextlib
from math import gamma

import numpy as np
import torch

from distributed_optimization_library.compressor import get_compressors, get_compressor_signatures, \
    CompressedVector, CompressedTorchVector, RandKCompressor, SumCompressedVectors, PermutationCompressor, \
    GZipVector
from distributed_optimization_library.transport import Transport, BroadcastNumpy, PartialParticipationError
from distributed_optimization_library.signature import Signature
from distributed_optimization_library.factory import Factory
from distributed_optimization_library.function import OptimizationProblemMeta, FunctionType

class FactoryMaster(Factory):
    pass

class FactoryNode(Factory):
    pass


def bernoulli_sample(random_generator, prob):
    if prob == 0.0:
        return False
    return random_generator.random() < prob


class BaseNodeAlgorithm(object):
    def __init__(self, function, seed=None, **kwargs):
        super(BaseNodeAlgorithm, self).__init__(**kwargs)
        self._function = function
        self._generator = np.random.default_rng(seed)
    
    def dim(self):
        return self._function.dim()
    
    def liptschitz_gradient_constant(self, *args, **kwargs):
        return self._function.liptschitz_gradient_constant(*args, **kwargs)
    
    def function_type(self):
        return self._function.type()
    
    def statistics(self):
        return self._function.statistics()


class BaseCompressorNodeAlgorithm(BaseNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(BaseCompressorNodeAlgorithm, self).__init__(function, **kwargs)
        self._compressor = compressor
    
    def num_nonzero_components(self):
        return self._compressor.num_nonzero_components()
    
    def omega(self):
        return self._compressor.omega()
    
    def compressor_biased(self):
        return self._compressor.biased()
    
    def compressor_independent(self):
        return self._compressor.independent()
    
    def compressor_random(self):
        return self._compressor.random()
    
    def compressor_factory_name(self):
        return self._compressor.factory_name()


class BaseMasterAlgorithm(object):
    def __init__(self, number_of_nodes, meta):
        self._meta = meta
        self._number_of_nodes = number_of_nodes
        assert self._number_of_nodes > 0
        self._iteration = 0
    
    def step(self):
        raise NotImplementedError()
    
    def get_point(self):
        raise NotImplementedError()
    
    def optimize(self, number_of_iterations):
        while self._iteration < number_of_iterations:
            self.step()
            self._iteration += 1
        return self.get_point()
    
    def stop(self):
        pass
    
    def _average(self, vectors, normalize=True):
        assert isinstance(vectors, list)
        average = sum(vectors)
        if normalize:
            average /= float(self._number_of_nodes)
        return average


class BaseTransportMasterAlgorithm(BaseMasterAlgorithm):
    def __init__(self, transport, meta):
        super(BaseTransportMasterAlgorithm, self).__init__(transport.get_number_of_nodes(), meta)
        self._transport = transport
    
    def get_stats(self):
        return self._transport.get_stat_to_nodes(), self._transport.get_stat_from_nodes()
    
    def get_max_stats(self):
        return self._transport.get_max_stat_from_nodes()
    
    def stop(self):
        self._transport.stop()
    
    def _average_compressed(self, vectors_compressed, number_of_nodes=None):
        number_of_nodes = number_of_nodes if number_of_nodes is not None else self._number_of_nodes
        assert isinstance(vectors_compressed, list)
        if isinstance(vectors_compressed[0], CompressedVector):
            average = np.zeros((vectors_compressed[0]._dim,), 
                               dtype=vectors_compressed[0]._values.dtype)
        elif isinstance(vectors_compressed[0], CompressedTorchVector):
            average = torch.zeros((vectors_compressed[0]._dim,),
                                  dtype=vectors_compressed[0]._values.dtype,
                                  device=vectors_compressed[0]._values.device)
        else:
            decompressed_vectors = sum([vector_compressed.decompress() 
                                        for vector_compressed in vectors_compressed])
            average = decompressed_vectors / float(number_of_nodes)
            return average
        for vector_compressed in vectors_compressed:
            average[vector_compressed._indices] += vector_compressed._values
        average = average / float(number_of_nodes)
        return average
    
    def statistics(self):
        statistics_nodes = self._transport.call_nodes_method(node_method="statistics")
        keys = statistics_nodes[0].keys()
        statistics = {}
        for key in keys:
            value = 0.0
            for statistics_node in statistics_nodes:
                value += statistics_node[key]
            value /= len(statistics_nodes)
            statistics[key] = value
        return statistics
    
    @contextlib.contextmanager
    def ignore_statistics(self):
        with self._transport.ignore_statistics():
            yield


class BaseCompressorMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, meta, compressor, **kwargs):
        super(BaseCompressorMasterAlgorithm, self).__init__(transport, meta, **kwargs)
        self._compressor = compressor


class BaseCompressorsMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, meta, compressors, **kwargs):
        super(BaseCompressorsMasterAlgorithm, self).__init__(transport, meta, **kwargs)
        self._compressors = compressors


# @TODO(tyurina): Delete this logic
class PointWithCachedGradient(object):
    def __init__(self, point, function):
        self._point = point
        self._function = function
        self._gradient = None
        self._value = None
    
    def get_point(self):
        return self._point

    def gradient(self):
        if not self._function.deterministic():
            return self._function.gradient(self._point)
        else:
            if self._gradient is None:
                self._gradient = self._function.gradient(self._point)
            return self._gradient

    def value(self):
        if not self._function.deterministic():
            return self._function.value(self._point)
        else:
            if self._value is None:
                self._value = self._function.value(self._point)
            return self._value


class BaseMarinaMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, meta, prob, seed):
        super(BaseMarinaMasterAlgorithm, self).__init__(transport, meta)
        self._prob = prob
        self._generator = np.random.default_rng(seed)
    
    def step(self):
        flag_calculate_gradient = bernoulli_sample(self._generator, self._prob)
        self._nodes_step()
        if flag_calculate_gradient:
            self._update_gradient_estimator_with_checkpoint()
        else:
            self._update_gradient_estimator()
    
    def get_point(self):
        return self._transport.call_node_method(node_index=0, node_method="get_point")
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient")
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function")
        return self._average(function_values)
    
    def _nodes_step(self):
        self._transport.call_nodes_method(node_method="step", gradient_estimator=self._gradient_estimator)
    
    def _calculate_checkpoint(self, *args, **kwargs):
        raise NotImplementedError()
    
    def _update_gradient_estimator_with_checkpoint(self, *args, **kwargs):
        checkpoint = self._calculate_checkpoint(*args, **kwargs)
        self._gradient_estimator = checkpoint
    
    def _update_gradient_estimator(self):
        updates = self._transport.call_nodes_method(node_method="calculate_updates")
        average_updates = self._average_compressed(updates)
        self._gradient_estimator = self._gradient_estimator + average_updates
    
    @staticmethod
    def _liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.mean(liptschitz_gradient_constants)
    
    @staticmethod
    def _tilde_liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.sqrt(np.mean(np.square(liptschitz_gradient_constants)))
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        return np.max(omegas)


@FactoryNode.register("marina_node")
class MarinaNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(MarinaNodeAlgorithm, self).__init__(function, compressor, **kwargs)
    
    def set_point(self, point):
        self._point = PointWithCachedGradient(point, self._function)
        
    def set_gamma(self, gamma):
        self._gamma = gamma
    
    def step(self, gradient_estimator):
        self._previous_point = self._point
        self._point = PointWithCachedGradient(
            self._point.get_point() - self._gamma * gradient_estimator,
            self._function)
    
    def calculate_gradient(self):
        return self._point.gradient()
    
    def calculate_gradient_at_previous_point(self):
        return self._previous_point.gradient()
    
    def calculate_function(self):
        return self._point.value()
    
    def calculate_updates(self):
        update_gradient_estimator = \
            self._compressor.compress(self._point.gradient() - self._previous_point.gradient())
        return update_gradient_estimator

    def get_point(self):
        return self._point.get_point()
    
    def get_previous_point(self):
        return self._previous_point.get_point()


@FactoryMaster.register("marina_master")
class MarinaMasterAlgorithm(BaseMarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, gamma_multiply=None, seed=None,
                 meta=None):
        prob = prob if prob is not None else self._estimate_prob(transport)
        super(MarinaMasterAlgorithm, self).__init__(transport, meta, prob, seed)
        gamma = gamma if gamma is not None else self._estimate_gamma()
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gradient_estimator = None
        self._gamma = gamma
        self._init(point, gamma)
    
    def _init(self, point, gamma):
        self._transport.call_nodes_method(node_method="set_point", point=point)
        self._transport.call_nodes_method(node_method="set_gamma", gamma=gamma)
        self._update_gradient_estimator_with_checkpoint()
    
    def get_previous_point(self):
        return self._transport.call_node_method(node_index=0, 
                                                node_method="get_previous_point")
    
    def _calculate_checkpoint(self):
        return self.calculate_gradient()
    
    @staticmethod
    def _estimate_prob(transport):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        return np.mean(num_nonzero_components) / dim
    
    def _estimate_gamma(self):
        assert np.all(np.logical_not(self._transport.call_nodes_method(node_method="compressor_biased")))
        assert np.all(self._transport.call_nodes_method(node_method="compressor_independent"))
        liptschitz_gradient_constant = self._liptschitz_gradient_constant(self._transport)
        tilde_liptschitz_gradient_constant = self._tilde_liptschitz_gradient_constant(self._transport)
        omega = self._omega(self._transport)
        inv_gamma = (liptschitz_gradient_constant + 
                     tilde_liptschitz_gradient_constant * 
                     np.sqrt(((1 - self._prob) * omega) / (self._prob * self._transport.get_number_of_nodes())))
        gamma = 1 / inv_gamma
        return gamma


@FactoryNode.register("marina_permutation_node")
class MarinaPermutationNodeAlgorithm(MarinaNodeAlgorithm):
    pass
    
    
@FactoryMaster.register("marina_permutation_master")
class MarinaPermutationMasterAlgorithm(MarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, seed=None, gamma_multiply=None, 
                 meta=None):
        prob = prob if prob is not None else self._estimate_prob(transport)
        gamma = gamma if gamma is not None else self._estimate_permutation_gamma(transport, meta, prob)
        super(MarinaPermutationMasterAlgorithm, self).__init__(transport=transport, point=point, gamma=gamma, 
                                                               gamma_multiply=gamma_multiply, 
                                                               prob=prob, seed=seed)
        
    @classmethod
    def _estimate_permutation_gamma(cls, transport, meta, prob):
        for factory_name in transport.call_nodes_method(node_method="compressor_factory_name"):
            assert factory_name == 'permutation' or factory_name == 'permutation_fixed_blocks'
        liptschitz_gradient_constant = cls._liptschitz_gradient_constant(transport)
        tilde_liptschitz_gradient_constant = cls._tilde_liptschitz_gradient_constant(transport)
        smoothness_variance = meta.smoothness_variance()
        smoothness_variance = smoothness_variance if smoothness_variance is not None else tilde_liptschitz_gradient_constant
        assert smoothness_variance <= tilde_liptschitz_gradient_constant
        omega = cls._omega(transport)
        inv_gamma = (liptschitz_gradient_constant + 
                     np.sqrt(((1 - prob) / prob) * 
                             (((omega + 1) / transport.get_number_of_nodes() - 1) * tilde_liptschitz_gradient_constant ** 2 + 
                              smoothness_variance ** 2)))
        gamma = 1 / inv_gamma
        return gamma
    
    def calculate_smoothness_variance(self):
        gradients = self._transport.call_nodes_method(
            node_method="calculate_gradient")
        previous_gradients = self._transport.call_nodes_method(
            node_method="calculate_gradient_at_previous_point")
        gradient = self._average(gradients)
        previous_gradient = self._average(previous_gradients)
        lambda_square_norm = lambda x: np.inner(x, x)
        square_norm_gradient = lambda_square_norm(gradient - previous_gradient)
        mean_square_norm_gradients = 0
        for gradient_node, previous_gradient_node in zip(gradients, previous_gradients):
            mean_square_norm_gradients += lambda_square_norm(gradient_node - previous_gradient_node)
        mean_square_norm_gradients /= self._number_of_nodes
        point = self.get_point()
        previous_point = self.get_previous_point()
        point_diff = lambda_square_norm(point - previous_point)
        return mean_square_norm_gradients / point_diff, square_norm_gradient / point_diff


@FactoryNode.register("marina_nodes_permutation_node")
class MarinaNodesPermutationNodeAlgorithm(MarinaNodeAlgorithm):
    pass
    
    
@FactoryMaster.register("marina_nodes_permutation_master")
class MarinaNodesPermutationMasterAlgorithm(MarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, seed=None, gamma_multiply=None, 
                 meta=None):
        prob = prob if prob is not None else self._estimate_prob(transport)
        gamma = gamma if gamma is not None else self._estimate_permutation_gamma(transport, meta, prob)
        super(MarinaNodesPermutationMasterAlgorithm, self).__init__(transport=transport, point=point, gamma=gamma, 
                                                                    gamma_multiply=gamma_multiply, 
                                                                    prob=prob, seed=seed)
        
    @classmethod
    def _estimate_permutation_gamma(cls, transport, meta, prob):
        for factory_name in transport.call_nodes_method(node_method="compressor_factory_name"):
            assert factory_name == 'nodes_permutation'
        liptschitz_gradient_constant = cls._liptschitz_gradient_constant(transport)
        tilde_liptschitz_gradient_constant = cls._tilde_liptschitz_gradient_constant(transport)
        smoothness_variance = meta.smoothness_variance()
        smoothness_variance = smoothness_variance if smoothness_variance is not None else tilde_liptschitz_gradient_constant
        assert smoothness_variance <= tilde_liptschitz_gradient_constant
        omega = cls._omega(transport)
        inv_gamma = (liptschitz_gradient_constant + 
                     smoothness_variance * np.sqrt(((1 - prob) / prob) * 
                                                   (omega / (transport.get_number_of_nodes() - 1))))
        gamma = 1 / inv_gamma
        return gamma


@FactoryNode.register("vr_marina_node")
class VRMarinaNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(VRMarinaNodeAlgorithm, self).__init__(function, compressor, **kwargs)
    
    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
    
    def set_point(self, point):
        self._point = point
        
    def set_gamma(self, gamma):
        self._gamma = gamma
    
    def step(self, gradient_estimator):
        self._previous_point = self._point
        self._point = self._point - self._gamma * gradient_estimator
    
    def calculate_gradient(self):
        return self._function.gradient(self._point)
    
    def calculate_gradient_statistics(self):
        return self._function.gradient_statistics(self._point)
    
    def calculate_function(self):
        return self._function.value(self._point)
    
    def number_of_functions(self):
        return self._function.number_of_functions()
    
    def calculate_updates(self):
        batch_gradient, previous_batch_gradient = self._function.batch_gradient_at_points_statistics(
            (self._point, self._previous_point), self._batch_size)
        update_gradient_estimator = \
            self._compressor.compress(batch_gradient - previous_batch_gradient)
        return update_gradient_estimator

    def get_point(self):
        return self._point
    
    def liptschitz_max_gradient_constant(self, *args, **kwargs):
        return self._function.liptschitz_max_gradient_constant(*args, **kwargs)


@FactoryMaster.register("vr_marina_master")
class VRMarinaMasterAlgorithm(BaseMarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, 
                 batch_size=None, gamma_multiply=None, seed=None,
                 meta=None):
        prob = prob if prob is not None else self._estimate_prob(transport, batch_size)
        gamma = gamma if gamma is not None else self._estimate_gamma(transport, prob, batch_size)
        super(VRMarinaMasterAlgorithm, self).__init__(transport, meta, prob, seed)
        self._batch_size = batch_size
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gradient_estimator = None
        self._gamma = gamma
        self._init(point, gamma, batch_size)
    
    def _init(self, point, gamma, batch_size):
        self._transport.call_nodes_method(node_method="set_point", point=point)
        self._transport.call_nodes_method(node_method="set_gamma", gamma=gamma)
        self._transport.call_nodes_method(node_method="set_batch_size", batch_size=batch_size)
        self._update_gradient_estimator_with_checkpoint()
    
    def _calculate_checkpoint(self):
        return self.calculate_gradient_statistics()
    
    def calculate_gradient_statistics(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient_statistics")
        return self._average(gradients)
    
    @staticmethod
    def _estimate_prob(transport, batch_size):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        number_of_functions = VRMarinaMasterAlgorithm._number_of_functions(transport)
        return np.minimum(float(batch_size) / (batch_size + number_of_functions),
                          np.mean(num_nonzero_components) / float(dim))
    
    @classmethod
    def _estimate_gamma(cls, transport, prob, batch_size):
        assert np.all(np.logical_not(transport.call_nodes_method(node_method="compressor_biased")))
        assert np.all(transport.call_nodes_method(node_method="compressor_independent"))
        liptschitz_gradient_constant = cls._liptschitz_gradient_constant(transport)
        tilde_liptschitz_gradient_constant = cls._tilde_liptschitz_gradient_constant(transport)
        tilde_max_liptschitz_gradient_constant = cls._tilde_max_liptschitz_gradient_constant(transport)
        omega = cls._omega(transport)
        inv_gamma = (liptschitz_gradient_constant + 
                     np.sqrt((omega * tilde_liptschitz_gradient_constant ** 2 + 
                              (omega + 1) * (tilde_max_liptschitz_gradient_constant ** 2) / batch_size)
                             * (1 - prob) / (prob * transport.get_number_of_nodes())))
        gamma = 1 / inv_gamma
        return gamma
    
    @staticmethod
    def _tilde_max_liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_max_gradient_constant")
        return np.sqrt(np.mean(np.square(liptschitz_gradient_constants)))
    
    @staticmethod
    def _number_of_functions(transport):
        numbers_of_functions = transport.call_nodes_method(node_method="number_of_functions")
        for lhs, rhs in zip(numbers_of_functions[1:], numbers_of_functions[:-1]):
            assert lhs == rhs, (lhs, rhs)
        return numbers_of_functions[0]


@FactoryNode.register("marina_stochastic_node")
class MarinaStochasticNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(MarinaStochasticNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def set_mega_batch_size(self, mega_batch_size):
        self._mega_batch_size = mega_batch_size
    
    def set_point(self, point):
        self._point = point
        
    def set_gamma(self, gamma):
        self._gamma = gamma
    
    def step(self, gradient_estimator):
        self._previous_point = self._point
        self._point = self._point - self._gamma * gradient_estimator
    
    def calculate_updates(self):
        stochastic_gradient, previous_stochastic_gradient = self._function.stochastic_gradient_at_points(
            (self._point, self._previous_point))
        update_gradient_estimator = \
            self._compressor.compress(stochastic_gradient - previous_stochastic_gradient)
        return update_gradient_estimator
    
    def calculate_mega_batch(self, mega_batch_size=None):
        mega_batch_size = self._mega_batch_size if mega_batch_size is None else mega_batch_size
        if mega_batch_size == 0:
            return zeros_based_on_function_type(len(self._point), self.function_type())
        mega_batch = 0.0
        for _ in range(mega_batch_size):
            mega_batch = mega_batch + self._function.stochastic_gradient(self._point)
        mega_batch = mega_batch / float(mega_batch_size)
        return mega_batch
    
    def calculate_gradient(self):
        return self._function.gradient(self._point)
    
    def calculate_function(self):
        return self._function.value(self._point)

    def get_point(self):
        return self._point


@FactoryMaster.register("marina_stochastic_master")
class MarinaStochasticMasterAlgorithm(BaseMarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, 
                 mega_batch_size=None, initial_mega_batch_size=None, 
                 gamma_multiply=None, seed=None, meta=None):
        prob = prob if prob is not None else self._estimate_prob(transport, mega_batch_size)
        super(MarinaStochasticMasterAlgorithm, self).__init__(transport, meta, prob, seed)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gradient_estimator = None
        self._gamma = gamma
        self._init(point, gamma, mega_batch_size, initial_mega_batch_size)
    
    def _init(self, point, gamma, mega_batch_size, initial_mega_batch_size):
        self._transport.call_nodes_method(node_method="set_point", point=point)
        self._transport.call_nodes_method(node_method="set_gamma", gamma=gamma)
        self._transport.call_nodes_method(node_method="set_mega_batch_size", mega_batch_size=mega_batch_size)
        self._update_gradient_estimator_with_checkpoint(mega_batch_size=initial_mega_batch_size)
    
    def _calculate_checkpoint(self, mega_batch_size=None):
        mega_batch_gradients = self._transport.call_nodes_method(node_method="calculate_mega_batch",
                                                                 mega_batch_size=mega_batch_size)
        return self._average(mega_batch_gradients)
    
    @staticmethod
    def _estimate_prob(transport, mega_batch_size):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        return np.minimum(1 / (1 + mega_batch_size),
                          np.mean(num_nonzero_components) / dim)


@FactoryNode.register("marina_partial_participation_node")
class MarinaPartialParticipationNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def init_node(self, point, gamma):
        gradient = self._function.gradient(point)
        self._gamma = gamma
        return gradient
    
    def step_checkpoint(self, point):
        return self._function.gradient(point)
        
    def step_compress(self, point, previous_point):
        message = self._function.gradient(point) - self._function.gradient(previous_point)
        compressed_message = self._compressor.compress(message)
        return compressed_message
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryNode.register("vr_marina_partial_participation_node")
class VRMarinaPartialParticipationNodeAlgorithm(MarinaPartialParticipationNodeAlgorithm):
    def set_batch_size(self, batch_size):
        self._batch_size = batch_size
    
    def step_compress(self, point, previous_point):
        batch_gradient, previous_batch_gradient = self._function.batch_gradient_at_points_statistics(
            (point, previous_point), self._batch_size)
        update_gradient_estimator = \
            self._compressor.compress(batch_gradient - previous_batch_gradient)
        return update_gradient_estimator

    def number_of_functions(self):
        return self._function.number_of_functions()


@FactoryNode.register("marina_stochastic_partial_participation_node")
class MarinaStochasticPartialParticipationNodeAlgorithm(MarinaPartialParticipationNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mega_batch_size = None
        self._initial_mega_batch_size = None
        
    def set_mega_batch_sizes(self, mega_batch_size, initial_mega_batch_size):
        self._mega_batch_size = mega_batch_size
        self._initial_mega_batch_size = initial_mega_batch_size
    
    def init_node(self, point, gamma):
        if self._initial_mega_batch_size is None:
            gradient = self._function.gradient(point)
        else:
            gradient = self.calculate_mega_batch(point, self._initial_mega_batch_size)
        self._gamma = gamma
        return gradient
    
    def step_checkpoint(self, point):
        return self.calculate_mega_batch(point, self._mega_batch_size)
    
    def step_compress(self, point, previous_point):
        stochastic_gradient, previous_stochastic_gradient = self._function.stochastic_gradient_at_points(
            (point, previous_point))
        update_gradient_estimator = \
            self._compressor.compress(stochastic_gradient - previous_stochastic_gradient)
        return update_gradient_estimator
    
    def calculate_mega_batch(self, point, mega_batch_size):
        if mega_batch_size == 0:
            return zeros_based_on_function_type(len(point), self.function_type())
        mega_batch = 0.0
        for _ in range(mega_batch_size):
            mega_batch = mega_batch + self._function.stochastic_gradient(point)
        mega_batch = mega_batch / float(mega_batch_size)
        return mega_batch

    def number_of_functions(self):
        return self._function.number_of_functions()


@FactoryMaster.register("marina_partial_participation_master")
class MarinaPartialParticipationMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 number_of_samples=None, prob=None,
                 mega_batch_size=None, initial_mega_batch_size=None,
                 seed=None, meta=None):
        super(MarinaPartialParticipationMasterAlgorithm, self).__init__(transport, meta)
        self._point = point
        self._number_of_samples = number_of_samples
        assert self._number_of_samples <= self._number_of_nodes
        self._prob = prob
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._generator = np.random.default_rng(seed)
        # @TODO: Fix it maybe. Not good
        if mega_batch_size is not None or initial_mega_batch_size is not None:
            self._transport.call_nodes_method(node_method="set_mega_batch_sizes", 
                                              mega_batch_size=mega_batch_size,
                                              initial_mega_batch_size=initial_mega_batch_size)
        initial_gradient_estimators = self._transport.call_nodes_method(
            node_method="init_node", point=point, gamma=self._gamma)
        self._gradient_estimator = self._average(initial_gradient_estimators)
        self._function_type = self._transport.call_node_method(node_method="function_type", node_index=0)
    
    def step(self):
        previous_point = self._point
        self._point = self._point - self._gamma * self._gradient_estimator
        point = self._point
        if self._function_type == FunctionType.NUMPY:
            point = BroadcastNumpy(point)
            previous_point = BroadcastNumpy(previous_point)
        calculate_checkpoint = bernoulli_sample(self._generator, self._prob)
        if calculate_checkpoint:
            gradients = self._transport.call_nodes_method(
                node_method="step_checkpoint", point=point)
            self._gradient_estimator = self._average(gradients)
        else:
            partial_participation_samples = self._generator.permutation(self._number_of_nodes)[:self._number_of_samples]
            partial_participation_samples = list(partial_participation_samples)
            compressed_messages = self._transport.call_nodes_method(
                node_method="step_compress", node_indices=partial_participation_samples,
                point=point, previous_point=previous_point)
            average_messages = self._average_compressed(compressed_messages, 
                                                        number_of_nodes=self._number_of_samples)
            self._gradient_estimator = self._gradient_estimator + average_messages
        
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=self._point)
        return self._average(function_values)
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]


@FactoryMaster.register("vr_marina_partial_participation_master")
class VRMarinaPartialParticipationMasterAlgorithm(MarinaPartialParticipationMasterAlgorithm):
    def __init__(self, *args, batch_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._transport.call_nodes_method(node_method="set_batch_size", batch_size=batch_size)
        self._prob = self._prob if self._prob is not None else \
            self._estimate_prob(self._transport, batch_size, self._number_of_samples, self._transport.get_number_of_nodes())
    
    @staticmethod
    def _estimate_prob(transport, batch_size, number_of_samples, number_of_nodes):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        number_of_functions = VRMarinaMasterAlgorithm._number_of_functions(transport)
        prob = np.minimum(float(batch_size) / (batch_size + number_of_functions),
                          np.mean(num_nonzero_components) / float(dim))
        prob *= (number_of_samples / number_of_nodes)
        return prob


@FactoryMaster.register("marina_stochastic_partial_participation_master")
class MarinaStochasticPartialParticipationMasterAlgorithm(MarinaPartialParticipationMasterAlgorithm):
    def __init__(self, *args, mega_batch_size=None, initial_mega_batch_size=None, **kwargs):
        super().__init__(*args, mega_batch_size=mega_batch_size, 
                         initial_mega_batch_size=initial_mega_batch_size, **kwargs)
        self._prob = self._prob if self._prob is not None else \
            self._estimate_prob(self._transport, mega_batch_size, self._number_of_samples, self._transport.get_number_of_nodes())
            
    @staticmethod
    def _estimate_prob(transport, mega_batch_size, number_of_samples, number_of_nodes):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        prob = np.minimum(1 / (1 + mega_batch_size),
                          np.mean(num_nonzero_components) / dim)
        prob *= (number_of_samples / number_of_nodes)
        return prob


def zeros_based_on_function_type(dim, function_type):
    if function_type == FunctionType.NUMPY:
        return np.zeros((dim,), dtype=np.float32)
    elif function_type == FunctionType.TORCH_CPU:
        return torch.zeros(dim)
    elif function_type == FunctionType.TORCH_CUDA:
        return torch.zeros(dim).cuda()
    else:
        raise RuntimeError()


def copy_numpy_or_torch(obj):
    if isinstance(obj, np.ndarray):
        return np.copy(obj)
    elif torch.is_tensor(obj):
        return obj.clone().detach()
    else:
        raise RuntimeError()


def norm_numpy_or_torch(vector):
    if isinstance(vector, np.ndarray):
        return np.linalg.norm(vector)
    elif torch.is_tensor(vector):
        return torch.norm(vector)
    else:
        raise RuntimeError()


class BaseZeroMarinaNodeAlgorithm(BaseCompressorNodeAlgorithm):
    
    def init_base_node(self, momentum):
        self._momentum = momentum
    
    def _calculate_local_gradient_estimators(point, *args, **kwargs):
        raise NotImplementedError()
    
    def step(self, point, *args, **kwargs):
        local_gradient_estimator, previous_local_gradient_estimator = \
            self._calculate_local_gradient_estimators(point, *args, **kwargs)
        message = local_gradient_estimator - previous_local_gradient_estimator - \
            self._momentum * (self._gradient_estimator - previous_local_gradient_estimator)
        compressed_message = self._compressor.compress(message)
        decompressed_message = compressed_message.decompress()
        self._gradient_estimator = self._gradient_estimator + decompressed_message
        return compressed_message
    
    def calculate_function(self):
        return self._function.value(self._point)
    
    def calculate_gradient(self):
        return self._function.gradient(self._point)


class BaseZeroMarinaSyncNodeAlgorithm(BaseZeroMarinaNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(BaseZeroMarinaSyncNodeAlgorithm, self).__init__(function, compressor, **kwargs)
    
    def _calculate_local_sync_gradient_estimators(point, *args, **kwargs):
        raise NotImplementedError()
    
    def step_sync(self, point, *args, **kwargs):
        local_gradient_estimator = self._calculate_local_sync_gradient_estimators(point, *args, **kwargs)
        self._stochastic_gradient_estimator = local_gradient_estimator
        self._gradient_estimator = local_gradient_estimator
        return local_gradient_estimator


@FactoryNode.register("zero_marina_node")
class ZeroMarinaNodeAlgorithm(BaseZeroMarinaNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(ZeroMarinaNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def init_node(self, point, momentum, init_with_gradients):
        self.init_base_node(momentum)
        dim = len(point)
        self._point = PointWithCachedGradient(point, self._function)
        if init_with_gradients:
            self._gradient_estimator = copy_numpy_or_torch(self._point.gradient())
            return self._gradient_estimator
        else:
            self._gradient_estimator = zeros_based_on_function_type(dim, self.function_type())
        
    def _calculate_local_gradient_estimators(self, point):
        self._previous_point = self._point
        self._point = PointWithCachedGradient(point, self._function)
        gradient = self._point.gradient()
        previous_gradient = self._previous_point.gradient()
        return gradient, previous_gradient
    
    def calculate_function(self):
        return self._point.value()
    
    def calculate_gradient(self):
        return self._point.gradient()


@FactoryNode.register("zero_marina_stochastic_node")
class ZeroMarinaStochasticNodeAlgorithm(BaseZeroMarinaNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(ZeroMarinaStochasticNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def init_node(self, point, momentum, noise_momentum, initial_mega_batch_size):
        self.init_base_node(momentum)
        self._noise_momentum = noise_momentum
        assert self._noise_momentum is not None
        self._point = point
        self._stochastic_gradient_estimator = self._calculate_mega_batch(initial_mega_batch_size)
        self._gradient_estimator = copy_numpy_or_torch(self._stochastic_gradient_estimator)
        return self._gradient_estimator
    
    def _calculate_mega_batch(self, mega_batch_size):
        if mega_batch_size == 0:
            return zeros_based_on_function_type(len(self._point), self.function_type())
        mega_batch = 0.0
        for _ in range(mega_batch_size):
            mega_batch = mega_batch + self._function.stochastic_gradient(self._point)
        mega_batch = mega_batch / float(mega_batch_size)
        return mega_batch
    
    def _calculate_local_gradient_estimators(self, point):
        self._previous_point = self._point
        self._point = point
        stochastic_gradient, previous_stochastic_gradient = \
            self._function.stochastic_gradient_at_points((self._point, self._previous_point))
        previous_stochastic_gradient_estimator = self._stochastic_gradient_estimator
        self._stochastic_gradient_estimator = stochastic_gradient + \
            (1 - self._noise_momentum) * (previous_stochastic_gradient_estimator - previous_stochastic_gradient)
        return self._stochastic_gradient_estimator, previous_stochastic_gradient_estimator


@FactoryNode.register("zero_marina_sync_stochastic_node")
class ZeroMarinaSyncStochasticNodeAlgorithm(BaseZeroMarinaSyncNodeAlgorithm, ZeroMarinaStochasticNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(ZeroMarinaSyncStochasticNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def init_node(self, point, momentum, mega_batch_size, initial_mega_batch_size):
        self._mega_batch_size = mega_batch_size
        return super(ZeroMarinaSyncStochasticNodeAlgorithm, self).init_node(
            point, momentum, noise_momentum=0.0, initial_mega_batch_size=initial_mega_batch_size)

    def _calculate_local_sync_gradient_estimators(self, point):
        self._previous_point = self._point
        self._point = point
        mega_stochastic_gradient = 0.0
        for _ in range(self._mega_batch_size):
            mega_stochastic_gradient += self._function.stochastic_gradient(self._point)
        mega_stochastic_gradient /= self._mega_batch_size
        return mega_stochastic_gradient


@FactoryNode.register("zero_marina_page_node")
class ZeroMarinaPageNodeAlgorithm(BaseZeroMarinaNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(ZeroMarinaPageNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def init_node(self, point, momentum, batch_size, independent_prob, prob):
        self.init_base_node(momentum)
        self._batch_size = batch_size
        self._point = point
        self._page_gradient_estimator = self._function.gradient(point)
        self._gradient_estimator = self._page_gradient_estimator
        self._independent_prob = independent_prob
        self._prob = prob
        return self._gradient_estimator
    
    def number_of_functions(self):
        return self._function.number_of_functions()
    
    def _calculate_local_gradient_estimators(self, point, flag_calculate_gradient):
        self._previous_point = self._point
        self._point = point
        previous_page_gradient_estimator = self._page_gradient_estimator
        if not self._independent_prob:
            assert flag_calculate_gradient is not None
        else:
            assert flag_calculate_gradient is None
            flag_calculate_gradient = bernoulli_sample(self._generator, self._prob)
        if flag_calculate_gradient:
            self._page_gradient_estimator = self._function.gradient(self._point)
        else:
            batch_gradient, previous_batch_gradient = self._function.batch_gradient_at_points(
                (self._point, self._previous_point), self._batch_size)
            self._page_gradient_estimator = previous_page_gradient_estimator + (batch_gradient - previous_batch_gradient)
        return self._page_gradient_estimator, previous_page_gradient_estimator


class BaseZeroMarinaMasterAlgorithm(BaseTransportMasterAlgorithm):
    def _init_base(self, point, gamma):
        self._point = point
        self._gamma = gamma
        self._function_type = self._transport.call_node_method(node_method="function_type", node_index=0)
    
    def _call_nodes_method(self, point):
        raise NotImplementedError()
    
    def step(self):
        self._point = self._point - self._gamma * self._gradient_estimator
        output_point = self._point
        if self._function_type == FunctionType.NUMPY:
            output_point = BroadcastNumpy(self._point)
        compressed_messages = self._call_nodes_method(output_point)
        average_updates = self._average_compressed(compressed_messages)
        self._gradient_estimator = self._gradient_estimator + average_updates
        
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient")
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function")
        return self._average(function_values)
    
    def _estimate_momentum(self):
        return 1.0 / (2 * self._omega(self._transport) + 1)
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]


class BaseZeroMarinaSyncMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, meta, seed):
        super(BaseZeroMarinaSyncMasterAlgorithm, self).__init__(transport, meta)
        self._generator = np.random.default_rng(seed)
    
    def _init_base(self, point, gamma, prob_sync):
        self._point = point
        self._gamma = gamma
        self._prob_sync = prob_sync
        self._function_type = self._transport.call_node_method(node_method="function_type", node_index=0)
    
    def _call_nodes_method(self, point):
        raise NotImplementedError()
    
    def _call_sync_nodes_method(self, point):
        raise NotImplementedError()
    
    def step(self):
        self._point = self._point - self._gamma * self._gradient_estimator
        output_point = self._point
        if self._function_type == FunctionType.NUMPY:
            output_point = BroadcastNumpy(self._point)
        flag_sync = bernoulli_sample(self._generator, self._prob_sync)
        if flag_sync:
            messages = self._call_sync_nodes_method(output_point)
            self._gradient_estimator = self._average(messages)
        else:
            compressed_messages = self._call_nodes_method(output_point)
            average_updates = self._average_compressed(compressed_messages)
            self._gradient_estimator = self._gradient_estimator + average_updates
        
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient")
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function")
        return self._average(function_values)
    
    def _estimate_momentum(self):
        return 1.0 / (2 * self._omega(self._transport) + 1)
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]


@FactoryMaster.register("zero_marina_master")
class ZeroMarinaMasterAlgorithm(BaseZeroMarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None, seed=None, init_with_gradients=True,
                 meta=None):
        super(ZeroMarinaMasterAlgorithm, self).__init__(transport, meta)
        momentum = momentum if momentum is not None else self._estimate_momentum()
        gamma = gamma if gamma is not None else self._estimate_gamma()
        self._momentum = momentum
        self._init_with_gradients = init_with_gradients
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._init(point, gamma, self._momentum)
    
    def _init(self, point, gamma, momentum):
        self._init_base(point, gamma)
        initial_gradient_estimator = self._transport.call_nodes_method(
            node_method="init_node", point=point, momentum=momentum,
            init_with_gradients=self._init_with_gradients)
        if self._init_with_gradients:
            self._gradient_estimator = self._average(initial_gradient_estimator)
        else:
            self._gradient_estimator = zeros_based_on_function_type(len(self._point), self._function_type)
    
    def _call_nodes_method(self, point):
        return self._transport.call_nodes_method(node_method="step", point=point)

    def _estimate_gamma(self):
        assert np.all(np.logical_not(self._transport.call_nodes_method(node_method="compressor_biased")))
        assert np.all(self._transport.call_nodes_method(node_method="compressor_independent"))
        liptschitz_gradient_constant = self._liptschitz_gradient_constant(self._transport)
        tilde_liptschitz_gradient_constant = self._tilde_liptschitz_gradient_constant(self._transport)
        omega = self._omega(self._transport)
        inv_gamma = (liptschitz_gradient_constant + 
                     tilde_liptschitz_gradient_constant * 
                     np.sqrt((omega * (4 * omega + 1)) / self._transport.get_number_of_nodes()))
        gamma = 1 / inv_gamma
        return gamma
    
    @staticmethod
    def _liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.mean(liptschitz_gradient_constants)
    
    @staticmethod
    def _tilde_liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.sqrt(np.mean(np.square(liptschitz_gradient_constants)))


@FactoryMaster.register("zero_marina_stochastic_master")
class ZeroMarinaStochasticMasterAlgorithm(BaseZeroMarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None, noise_momentum=None, initial_mega_batch_size=None,
                 seed=None, meta=None):
        super(ZeroMarinaStochasticMasterAlgorithm, self).__init__(transport, meta)
        momentum = momentum if momentum is not None else self._estimate_momentum()
        self._momentum = momentum
        assert noise_momentum is not None
        self._noise_momentum = noise_momentum
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._init(point, gamma, self._momentum, self._noise_momentum, initial_mega_batch_size)
    
    def _init(self, point, gamma, momentum, noise_momentum, initial_mega_batch_size):
        self._init_base(point, gamma)
        initial_gradient_estimator = self._transport.call_nodes_method(
            node_method="init_node", point=point, momentum=momentum, noise_momentum=noise_momentum,
            initial_mega_batch_size=initial_mega_batch_size)
        self._gradient_estimator = self._average(initial_gradient_estimator)
        
    def _call_nodes_method(self, point):
        return self._transport.call_nodes_method(node_method="step", point=point)


@FactoryMaster.register("zero_marina_sync_stochastic_master")
class ZeroMarinaSyncStochasticMasterAlgorithm(BaseZeroMarinaSyncMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None, prob_sync=None, mega_batch_size=None, initial_mega_batch_size=None,
                 seed=None, meta=None):
        super(ZeroMarinaSyncStochasticMasterAlgorithm, self).__init__(transport, meta, seed)
        momentum = momentum if momentum is not None else self._estimate_momentum()
        self._momentum = momentum
        assert prob_sync > 0.0
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._init(point, gamma, self._momentum, prob_sync, mega_batch_size, initial_mega_batch_size)
    
    def _init(self, point, gamma, momentum, prob_sync, mega_batch_size, initial_mega_batch_size):
        self._init_base(point, gamma, prob_sync)
        initial_gradient_estimators = self._transport.call_nodes_method(
            node_method="init_node", point=point, momentum=momentum,
            mega_batch_size=mega_batch_size, initial_mega_batch_size=initial_mega_batch_size)
        self._gradient_estimator = self._average(initial_gradient_estimators)
        
    def _call_nodes_method(self, point):
        return self._transport.call_nodes_method(node_method="step", point=point)
    
    def _call_sync_nodes_method(self, point):
        return self._transport.call_nodes_method(node_method="step_sync", point=point)


@FactoryMaster.register("zero_marina_page_master")
class ZeroMarinaPageMasterAlgorithm(BaseZeroMarinaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None, batch_size=None, prob=None,
                 independent_probabilistic_switching=False,
                 seed=None, meta=None):
        super(ZeroMarinaPageMasterAlgorithm, self).__init__(transport, meta)
        momentum = momentum if momentum is not None else self._estimate_momentum()
        prob = prob if prob is not None else self._estimate_prob(batch_size)
        self._independent_probabilistic_switching = independent_probabilistic_switching
        
        self._generator = np.random.default_rng(seed)
        self._momentum = momentum
        self._prob = prob
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._batch_size = batch_size
        self._init(point, gamma, self._momentum, self._batch_size,
                   independent_probabilistic_switching, self._prob)
    
    def _init(self, point, gamma, momentum, batch_size, independent_prob, prob):
        self._init_base(point, gamma)
        gradient_estimators = self._transport.call_nodes_method(node_method="init_node", 
                                                                point=point, momentum=momentum,
                                                                batch_size=batch_size,
                                                                independent_prob=independent_prob,
                                                                prob=prob)
        self._gradient_estimator = self._average(gradient_estimators)
        
    def _call_nodes_method(self, point):
        flag_calculate_gradient = None
        if not self._independent_probabilistic_switching:
            flag_calculate_gradient = bernoulli_sample(self._generator, self._prob)
        return self._transport.call_nodes_method(
            node_method="step", point=point, flag_calculate_gradient=flag_calculate_gradient)
    
    def _estimate_prob(self, batch_size):
        numbers_of_functions = self._number_of_functions(self._transport)
        return float(batch_size) / (batch_size + numbers_of_functions)
    
    @staticmethod
    def _number_of_functions(transport):
        numbers_of_functions = transport.call_nodes_method(node_method="number_of_functions")
        for lhs, rhs in zip(numbers_of_functions[1:], numbers_of_functions[:-1]):
            assert lhs == rhs
        return numbers_of_functions[0]


class BaseZeroMarinaPartialParticipationNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def init_node(self, momentum, noise_momentum, partial_participation_probability, 
                  return_gradient_estimator, *args, **kwargs):
        self._momentum = momentum
        self._noise_momentum = noise_momentum
        self._partial_participation_probability = partial_participation_probability
        self._stochastic_gradient_estimator, self._gradient_estimator = self._init_estimators(*args, **kwargs)
        if return_gradient_estimator:
            return self._gradient_estimator

    def _init_estimators(self, *args, **kwargs):
        raise NotImplementedError()
    
    def _calculate_gradient_difference(self, point, previous_point, *args, **kwargs):
        raise NotImplementedError()
    
    def step(self, point, previous_point, *args, **kwargs):
        gradient_difference = self._calculate_gradient_difference(point, previous_point, *args, **kwargs)
        previous_stochastic_gradient_estimator = self._stochastic_gradient_estimator
        self._stochastic_gradient_estimator = self._stochastic_gradient_estimator + \
            (1 / self._partial_participation_probability) * gradient_difference
        message = gradient_difference -\
            self._momentum * (self._gradient_estimator - previous_stochastic_gradient_estimator)
        compressed_message = self._compressor.compress((1 / self._partial_participation_probability) * message)
        decompressed_message = compressed_message.decompress()
        self._gradient_estimator = self._gradient_estimator + decompressed_message
        return compressed_message
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryNode.register("zero_marina_partial_participation_node")
class ZeroMarinaPartialParticipationNodeAlgorithm(BaseZeroMarinaPartialParticipationNodeAlgorithm):
    def _init_estimators(self, point, init_with_gradients):
        if init_with_gradients:
            stochastic_gradient_estimator = self._function.gradient(point)
            gradient_estimator = copy_numpy_or_torch(stochastic_gradient_estimator)
        else:
            stochastic_gradient_estimator = self._function.gradient(point)
            gradient_estimator = zeros_based_on_function_type(len(point), self.function_type())
        return stochastic_gradient_estimator, gradient_estimator
    
    def _calculate_gradient_difference(self, point, previous_point):
        gradient = self._function.gradient(point)
        previous_gradient = self._function.gradient(previous_point)
        gradient_difference = gradient - previous_gradient - \
            self._noise_momentum * (self._stochastic_gradient_estimator - previous_gradient)
        return gradient_difference


@FactoryNode.register("zero_marina_partial_participation_page_node")
class ZeroMarinaPartialParticipationPageNodeAlgorithm(BaseZeroMarinaPartialParticipationNodeAlgorithm):
    def init_node(self, batch_size, *args, **kwargs):
        self._batch_size = batch_size
        return super(ZeroMarinaPartialParticipationPageNodeAlgorithm, self).init_node(*args, **kwargs)
    
    def _init_estimators(self, point):
        stochastic_gradient_estimator = self._function.gradient(point)
        gradient_estimator = copy_numpy_or_torch(stochastic_gradient_estimator)
        return stochastic_gradient_estimator, gradient_estimator
    
    def _calculate_gradient_difference(self, point, previous_point, flag_calculate_gradient):
        if flag_calculate_gradient:
            gradient = self._function.gradient(point)
            previous_gradient = self._function.gradient(previous_point)
            gradient_difference = gradient - previous_gradient - \
                self._noise_momentum * (self._stochastic_gradient_estimator - previous_gradient)
        else:
            batch_gradient, previous_batch_gradient = self._function.batch_gradient_at_points(
                (point, previous_point), self._batch_size)
            gradient_difference = batch_gradient - previous_batch_gradient
        return gradient_difference

    def number_of_functions(self):
        return self._function.number_of_functions()


@FactoryNode.register("zero_marina_partial_participation_stochastic_node")
class ZeroMarinaStochasticPartialParticipationNodeAlgorithm(BaseZeroMarinaPartialParticipationNodeAlgorithm):
    def _init_estimators(self, point, initial_mega_batch_size):
        stochastic_gradient_estimator = self._calculate_mega_batch(initial_mega_batch_size, point)
        gradient_estimator = copy_numpy_or_torch(stochastic_gradient_estimator)
        return stochastic_gradient_estimator, gradient_estimator
    
    def _calculate_mega_batch(self, mega_batch_size, point):
        if mega_batch_size == 0:
            return zeros_based_on_function_type(len(point), self.function_type())
        mega_batch = 0.0
        for _ in range(mega_batch_size):
            mega_batch = mega_batch + self._function.stochastic_gradient(point)
        mega_batch = mega_batch / float(mega_batch_size)
        return mega_batch
    
    def _calculate_gradient_difference(self, point, previous_point):
        stochastic_gradient, previous_stochastic_gradient = \
            self._function.stochastic_gradient_at_points((point, previous_point))
        gradient_difference = stochastic_gradient - previous_stochastic_gradient - \
            self._noise_momentum * (self._stochastic_gradient_estimator - previous_stochastic_gradient)
        return gradient_difference


class BaseFreconNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(BaseFreconNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        assert self._compressor.independent()
        self._copy_compressor = compressor.copy()
        
    def _calculate_init_estimate(self, initial_mega_batch_size, point):
        raise NotImplementedError()
    
    def init_node(self, point, alpha, initial_mega_batch_size):
        self._alpha = alpha
        self._help_gradient_estimator = self._calculate_init_estimate(initial_mega_batch_size, point)
        return self._help_gradient_estimator
    
    def _local_gradient_estimates(self, point, previous_point):
        raise NotImplementedError()
    
    def step(self, point, previous_point):
        stochastic_gradient, previous_stochastic_gradient = self._local_gradient_estimates(point, previous_point)
        compressed_message_q = self._compressor.compress(stochastic_gradient - previous_stochastic_gradient)
        compressed_message_u = self._copy_compressor.compress(previous_stochastic_gradient - self._help_gradient_estimator)
        message_u = compressed_message_u.decompress()
        self._help_gradient_estimator = self._help_gradient_estimator + self._alpha * message_u
        return compressed_message_q, compressed_message_u
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryNode.register("frecon_node")
class FreconNodeAlgorithm(BaseFreconNodeAlgorithm):
    def _local_gradient_estimates(self, point, previous_point):
        gradient = self._function.gradient(point)
        previous_gradient = self._function.gradient(previous_point)
        return gradient, previous_gradient
    
    def _calculate_init_estimate(self, initial_mega_batch_size, point):
        assert initial_mega_batch_size is None
        return self._function.gradient(point)


@FactoryNode.register("frecon_stochastic_node")
class FreconStochasticNodeAlgorithm(BaseFreconNodeAlgorithm):
    def _local_gradient_estimates(self, point, previous_point):
        stochastic_gradient, previous_stochastic_gradient = self._function.stochastic_gradient_at_points(
            (point, previous_point))
        return stochastic_gradient, previous_stochastic_gradient
    
    def _calculate_init_estimate(self, initial_mega_batch_size, point):
        return self._calculate_mega_batch(initial_mega_batch_size, point)
    
    def _calculate_mega_batch(self, mega_batch_size, point):
        if mega_batch_size == 0:
            return zeros_based_on_function_type(len(point), self.function_type())
        mega_batch = 0.0
        for _ in range(mega_batch_size):
            mega_batch = mega_batch + self._function.stochastic_gradient(point)
        mega_batch = mega_batch / float(mega_batch_size)
        return mega_batch


class BaseZeroMarinaPartialParticipationMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, meta, number_of_samples, seed):
        super(BaseZeroMarinaPartialParticipationMasterAlgorithm, self).__init__(transport, meta)
        self._number_of_samples = number_of_samples
        self._partial_participation_probability = self._estimate_partial_participation_probability(
            number_of_samples, transport)
        self._generator = np.random.default_rng(seed)
    
    def _init_base(self, point, gamma):
        self._point = point
        self._gamma = gamma
        self._function_type = self._transport.call_node_method(node_method="function_type", node_index=0)
    
    def step(self):
        previous_point = self._point
        self._point = self._point - self._gamma * self._gradient_estimator
        point = self._point
        if self._function_type == FunctionType.NUMPY:
            point = BroadcastNumpy(point)
            previous_point = BroadcastNumpy(previous_point)
        partial_participation_samples = self._generator.permutation(self._number_of_nodes)[:self._number_of_samples]
        partial_participation_samples = list(partial_participation_samples)
        additional_arguments = self._additional_arguments()
        compressed_messages = self._transport.call_nodes_method(
            node_method="step", node_indices=partial_participation_samples,
            point=point, previous_point=previous_point, 
            **additional_arguments)
        average_updates = self._average_compressed(compressed_messages)
        self._gradient_estimator = self._gradient_estimator + average_updates
    
    def _additional_arguments(self):
        return {}
        
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=self._point)
        return self._average(function_values)
    
    @staticmethod
    def _estimate_momentum(number_of_samples, transport):
        partial_participation_probability = \
            BaseZeroMarinaPartialParticipationMasterAlgorithm._estimate_partial_participation_probability(
                number_of_samples, transport)
        omega = BaseZeroMarinaPartialParticipationMasterAlgorithm._omega(transport)
        return partial_participation_probability / (2 * omega + 1)
    
    @staticmethod
    def _estimate_partial_participation_probability(number_of_samples, transport):
        return number_of_samples / float(transport.get_number_of_nodes())
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]


@FactoryMaster.register("zero_marina_partial_participation_master")
class ZeroMarinaPartialParticipationMasterAlgorithm(BaseZeroMarinaPartialParticipationMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None,
                 number_of_samples=None,
                 init_with_gradients=True,
                 seed=None, meta=None):
        super(ZeroMarinaPartialParticipationMasterAlgorithm, self).__init__(
            transport, meta, number_of_samples, seed)
        momentum = momentum if momentum is not None else self._estimate_momentum(number_of_samples, self._transport)
        self._momentum = momentum
        self._init_with_gradients = init_with_gradients
        noise_momentum = self._partial_participation_probability / (2 - self._partial_participation_probability)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._init(point, gamma, self._momentum, noise_momentum)
    
    def _init(self, point, gamma, momentum, noise_momentum):
        self._init_base(point, gamma)
        gradients = self._transport.call_nodes_method(
            node_method="init_node", 
            momentum=momentum, noise_momentum=noise_momentum,
            partial_participation_probability=self._partial_participation_probability,
            return_gradient_estimator=self._init_with_gradients,
            init_with_gradients=self._init_with_gradients,
            point=point)
        if self._init_with_gradients:
            self._gradient_estimator = self._average(gradients)
        else:
            function_type = self._transport.call_node_method(node_method="function_type", node_index=0)
            self._gradient_estimator = zeros_based_on_function_type(len(self._point), function_type)


@FactoryMaster.register("zero_marina_partial_participation_page_master")
class ZeroMarinaPartialParticipationPageMasterAlgorithm(BaseZeroMarinaPartialParticipationMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None, batch_size=None,
                 number_of_samples=None,
                 seed=None, meta=None):
        super(ZeroMarinaPartialParticipationPageMasterAlgorithm, self).__init__(
            transport, meta, number_of_samples, seed)
        momentum = momentum if momentum is not None else self._estimate_momentum(number_of_samples, self._transport)
        self._momentum = momentum
        self._prob = self._estimate_prob(batch_size)
        noise_momentum = self._partial_participation_probability / (2 - self._partial_participation_probability)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._init(point, gamma, self._momentum, noise_momentum, batch_size)
    
    def _init(self, point, gamma, momentum, noise_momentum, batch_size):
        self._init_base(point, gamma)
        initial_gradient_estimator = self._transport.call_nodes_method(
            node_method="init_node", momentum=momentum, noise_momentum=noise_momentum,
            partial_participation_probability=self._partial_participation_probability,
            return_gradient_estimator=True,
            point=point, batch_size=batch_size)
        self._gradient_estimator = self._average(initial_gradient_estimator)
    
    def _estimate_prob(self, batch_size):
        numbers_of_functions = self._number_of_functions(self._transport)
        return float(batch_size) / (batch_size + numbers_of_functions)
    
    def _additional_arguments(self):
        flag_calculate_gradient = bernoulli_sample(self._generator, self._prob)
        return {'flag_calculate_gradient': flag_calculate_gradient}
    
    @staticmethod
    def _number_of_functions(transport):
        numbers_of_functions = transport.call_nodes_method(node_method="number_of_functions")
        for lhs, rhs in zip(numbers_of_functions[1:], numbers_of_functions[:-1]):
            assert lhs == rhs
        return numbers_of_functions[0]


@FactoryMaster.register("zero_marina_partial_participation_stochastic_master")
class ZeroMarinaPartialParticipationStochasticMasterAlgorithm(BaseZeroMarinaPartialParticipationMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 momentum=None, noise_momentum=None, 
                 number_of_samples=None, initial_mega_batch_size=None,
                 seed=None, meta=None):
        super(ZeroMarinaPartialParticipationStochasticMasterAlgorithm, self).__init__(
            transport, meta, number_of_samples, seed)
        momentum = momentum if momentum is not None else self._estimate_momentum(number_of_samples, self._transport)
        self._momentum = momentum
        assert noise_momentum is not None
        self._noise_momentum = noise_momentum
        partial_participation_probability = number_of_samples / self._number_of_nodes
        assert noise_momentum <= partial_participation_probability / (2 - partial_participation_probability)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._init(point, gamma, self._momentum, self._noise_momentum, initial_mega_batch_size)
    
    def _init(self, point, gamma, momentum, noise_momentum, initial_mega_batch_size):
        self._init_base(point, gamma)
        initial_gradient_estimator = self._transport.call_nodes_method(
            node_method="init_node", momentum=momentum, noise_momentum=noise_momentum,
            partial_participation_probability=self._partial_participation_probability,
            return_gradient_estimator=True,
            point=point, initial_mega_batch_size=initial_mega_batch_size)
        self._gradient_estimator = self._average(initial_gradient_estimator)


class BaseFreconMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None,
                 number_of_samples=None, initial_mega_batch_size=None,
                 seed=None, meta=None):
        super(BaseFreconMasterAlgorithm, self).__init__(transport, meta)
        self._number_of_samples = number_of_samples
        assert self._number_of_samples <= self._number_of_nodes
        self._alpha = self._estimate_alpha(self._transport)
        self._lambda = self._estimate_lambda()
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._point = point
        self._gamma = gamma
        self._generator = np.random.default_rng(seed)
        initial_gradient_estimator = self._transport.call_nodes_method(
            node_method="init_node", point=point, alpha=self._alpha,
            initial_mega_batch_size=initial_mega_batch_size)
        self._function_type = self._transport.call_node_method(node_method="function_type", node_index=0)
        self._gradient_estimator = zeros_based_on_function_type(len(self._point), self._function_type)
        self._help_gradient_estimator = self._average(initial_gradient_estimator)
    
    def step(self):
        previous_point = self._point
        self._point = self._point - self._gamma * self._gradient_estimator
        point = self._point
        if self._function_type == FunctionType.NUMPY:
            point = BroadcastNumpy(point)
            previous_point = BroadcastNumpy(previous_point)
        partial_participation_samples = self._generator.permutation(self._number_of_nodes)[:self._number_of_samples]
        partial_participation_samples = list(partial_participation_samples)
        compressed_messages = self._transport.call_nodes_method(
            node_method="step", node_indices=partial_participation_samples,
            point=point, previous_point=previous_point)
        compressed_messages_q = [compressed_message[0] for compressed_message in compressed_messages]
        compressed_messages_u = [compressed_message[1] for compressed_message in compressed_messages]
        averaga_q = self._average_compressed(compressed_messages_q, number_of_nodes=self._number_of_samples)
        averaga_u = self._average_compressed(compressed_messages_u, number_of_nodes=self._number_of_samples)
        self._gradient_estimator = averaga_q + (1 - self._lambda) * self._gradient_estimator + \
            self._lambda * (averaga_u + self._help_gradient_estimator)
        self._help_gradient_estimator = self._help_gradient_estimator + \
            (self._number_of_samples * self._alpha / self._number_of_nodes) * averaga_u
        
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=self._point)
        return self._average(function_values)
    
    @classmethod
    def _estimate_alpha(cls, transport):
        return 1. / (1. + cls._omega(transport))
    
    def _estimate_lambda(self):
        return self._number_of_samples / (2 * (1 + self._omega(self._transport)) * self._number_of_nodes)
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]


@FactoryMaster.register("frecon_master")
class FreconMasterAlgorithm(BaseFreconMasterAlgorithm):
    pass


@FactoryMaster.register("frecon_stochastic_master")
class FreconStochasticMasterAlgorithm(BaseFreconMasterAlgorithm):
    pass


@FactoryNode.register("gradient_descent_node")
class GradientDescentNodeAlgorithm(BaseNodeAlgorithm):
    def __init__(self, function, **kwargs):
        super(GradientDescentNodeAlgorithm, self).__init__(function, **kwargs)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("gradient_descent_master")
class GradientDescentMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(GradientDescentMasterAlgorithm, self).__init__(transport, meta)
        self._point = point
        self._gamma = gamma if gamma is not None else self._estimate_gamma()
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
    
    def step(self):
        self._point = self._point - self._gamma * self.calculate_gradient()

    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=self._point)
        return self._average(function_values)
    
    def _estimate_gamma(self):
        liptschitz_gradient_constant = self._liptschitz_gradient_constant(self._transport)
        gamma = 1 / liptschitz_gradient_constant
        return gamma
    
    @staticmethod
    def _liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.mean(liptschitz_gradient_constants)


@FactoryNode.register("stochastic_gradient_descent_node")
class StochasticGradientDescentNodeAlgorithm(BaseNodeAlgorithm):
    def __init__(self, function, **kwargs):
        super(StochasticGradientDescentNodeAlgorithm, self).__init__(function, **kwargs)
    
    def calculate_stochastic_gradient(self, point):
        return self._function.stochastic_gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryMaster.register("stochastic_gradient_descent_master")
class StochasticGradientDescentMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(StochasticGradientDescentMasterAlgorithm, self).__init__(transport, meta)
        self._point = point
        self._gamma = gamma
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
    
    def step(self):
        self._point = self._point - self._gamma * self.calculate_stochastic_gradient()

    def get_point(self):
        return self._point
    
    def calculate_stochastic_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_stochastic_gradient",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_function",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=self._point)
        return self._average(gradients)


@FactoryNode.register("rand_diana_node")
class RandDianaNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(RandDianaNodeAlgorithm, self).__init__(*args, **kwargs)
        self._local_shift = None
        self._point = None
    
    def _init_shifts(self):
        self._local_shift = zeros_based_on_function_type(
            len(self._point), self.function_type())
        
    def _set_point(self, point):
        self._point = point
    
    def _calculate_message(self):
        gradient = self._function.gradient(self._point)
        shifted_local_gradient = self._compressor.compress(
            gradient - self._local_shift)
        return shifted_local_gradient
    
    def _calculate_shift(self):
        gradient = self._function.gradient(self._point)
        self._local_shift = gradient
        return self._local_shift
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("rand_diana_master")
class RandDianaMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, gamma_multiply=None, seed=None,
                 meta=None, strong_convex_regime=False,
                 skip_initial_init=False):
        super(RandDianaMasterAlgorithm, self).__init__(transport, meta)
        self._generator = np.random.default_rng(seed)
        self._prob = prob if prob is not None else self._estimate_prob(transport)
        self._gamma = gamma if gamma is not None else self._estimate_gamma(self._meta, strong_convex_regime)
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._point = point
        self._skip_initial_init = skip_initial_init
        self._init(point)
    
    def _init(self, point):
        self._function_type = self._transport.call_node_method(
            node_method="function_type", node_index=0)
        output_point = point
        if self._function_type == FunctionType.NUMPY:
            output_point = BroadcastNumpy(output_point)
        self._transport.call_nodes_method(node_method="_set_point", point=output_point)
        self._transport.call_nodes_method(node_method="_init_shifts")
        self._shift = zeros_based_on_function_type(len(point), self._function_type)
        if not self._skip_initial_init:
            self._update_shifts()
    
    def step(self):
        output_point = self._point
        if self._function_type == FunctionType.NUMPY:
            output_point = BroadcastNumpy(output_point)
        self._transport.call_nodes_method(node_method="_set_point", point=output_point)
        messages = self._transport.call_nodes_method(node_method="_calculate_message")
        message = self._average_compressed(messages)
        gradient_estimator = message + self._shift
        self._point = self._point - self._gamma * gradient_estimator
        flag_calculate_gradient = bernoulli_sample(self._generator, self._prob)
        if flag_calculate_gradient:
            self._update_shifts()
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        output_point = self._point
        if self._function_type == FunctionType.NUMPY:
            output_point = BroadcastNumpy(output_point)
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=output_point)
        return self._average(gradients)
    
    def calculate_function(self):
        output_point = self._point
        if self._function_type == FunctionType.NUMPY:
            output_point = BroadcastNumpy(output_point)
        function_values = self._transport.call_nodes_method(node_method="calculate_function", 
                                                            point=output_point)
        return self._average(function_values)
    
    def _update_shifts(self):
        shifts = self._transport.call_nodes_method(node_method="_calculate_shift")
        self._shift = self._average(shifts)
        
    @staticmethod
    def _estimate_prob(transport):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        return np.mean(num_nonzero_components) / dim
    
    def _estimate_gamma(self, meta, strong_convex_regime):
        assert np.all(np.logical_not(self._transport.call_nodes_method(node_method="compressor_biased")))
        assert np.all(self._transport.call_nodes_method(node_method="compressor_independent"))
        omega = self._omega(self._transport)
        assert omega + 1 == self._transport._number_of_nodes, (omega + 1, self._transport._number_of_nodes)
        liptschitz_gradient_constant, smoothness_variance, bregman_smoothness_variance = \
            self._liptschitz_gradient_constants(self._transport, meta)
        if not strong_convex_regime:
            gamma = 1 / (4 * (5 * liptschitz_gradient_constant + 4 * bregman_smoothness_variance))
            gamma = min(gamma, np.sqrt(self._prob) / (2 * np.sqrt(smoothness_variance ** 2 + liptschitz_gradient_constant ** 2)))
        else:
            strong_convexity_constant = meta.strongly_convex_constant()
            gamma = 1 / (7 * liptschitz_gradient_constant + 6 * bregman_smoothness_variance)
            gamma = min(gamma, self._prob / (2 * strong_convexity_constant))
        return gamma
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        return np.max(omegas)
    
    @staticmethod
    def _tilde_liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.sqrt(np.mean(np.square(liptschitz_gradient_constants)))
    
    @staticmethod
    def _liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.mean(liptschitz_gradient_constants)
    
    @staticmethod
    def _max_liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.max(liptschitz_gradient_constants)
    
    @classmethod
    def _liptschitz_gradient_constants(cls, transport, meta):
        liptschitz_gradient_constant = cls._liptschitz_gradient_constant(transport)
        tilde_liptschitz_gradient_constant = cls._tilde_liptschitz_gradient_constant(transport)
        smoothness_variance = meta.smoothness_variance()
        smoothness_variance = smoothness_variance \
            if smoothness_variance is not None else tilde_liptschitz_gradient_constant
        assert smoothness_variance <= tilde_liptschitz_gradient_constant
        max_liptschitz_gradient_constant = cls._max_liptschitz_gradient_constant(transport)
        bregman_smoothness_variance = meta.bregman_smoothness_variance()
        bregman_smoothness_variance = bregman_smoothness_variance \
            if bregman_smoothness_variance is not None else max_liptschitz_gradient_constant
        assert bregman_smoothness_variance <= max_liptschitz_gradient_constant
        return (liptschitz_gradient_constant, smoothness_variance, bregman_smoothness_variance)


@FactoryNode.register("rand_diana_permutation_node")
class RandDianaPermutationNodeAlgorithm(RandDianaNodeAlgorithm):
    pass
    
    
@FactoryMaster.register("rand_diana_permutation_master")
class RandDianaPermutationMasterAlgorithm(RandDianaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, prob=None, gamma_multiply=None, seed=None,
                 meta=None, strong_convex_regime=False):
        prob = prob if prob is not None else self._estimate_prob(transport)
        gamma = gamma if gamma is not None else self._estimate_permutation_gamma(
            transport, prob, meta, strong_convex_regime)
        super(RandDianaPermutationMasterAlgorithm, self).__init__(transport=transport, point=point, gamma=gamma, 
                                                                  gamma_multiply=gamma_multiply, 
                                                                  prob=prob, seed=seed,
                                                                  meta=meta)
        
    @classmethod
    def _estimate_permutation_gamma(cls, transport, prob, meta, strong_convex_regime):
        for factory_name in transport.call_nodes_method(node_method="compressor_factory_name"):
            assert factory_name == 'permutation'
        liptschitz_gradient_constant, smoothness_variance, bregman_smoothness_variance = \
            cls._liptschitz_gradient_constants(transport, meta)
        if not strong_convex_regime:
            gamma = 1 / (4 * (liptschitz_gradient_constant + 4 * bregman_smoothness_variance))
            if smoothness_variance > 0:
                gamma = min(gamma, np.sqrt(prob) / (2 * smoothness_variance))
        else:
            strong_convexity_constant = meta.strongly_convex_constant()
            gamma = 1 / (liptschitz_gradient_constant + 6 * bregman_smoothness_variance)
            gamma = min(gamma, prob / (2 * strong_convexity_constant))
        return gamma


@FactoryNode.register("dcgd_node")
class DCGDNodeAlgorithm(RandDianaNodeAlgorithm):
    pass


@FactoryMaster.register("dcgd_master")
class DCGDMasterAlgorithm(RandDianaMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        gamma = gamma if gamma is not None else self._estimate_gamma(transport, meta)
        super(DCGDMasterAlgorithm, self).__init__(transport=transport, point=point, gamma=gamma, 
                                                  gamma_multiply=gamma_multiply, 
                                                  prob=0.0, skip_initial_init=True, seed=seed,
                                                  meta=meta)
        
    @classmethod
    def _estimate_gamma(cls, transport, meta):
        liptschitz_gradient_constant, _, bregman_smoothness_variance = \
            cls._liptschitz_gradient_constants(transport, meta)
        omega = cls._omega(transport)
        number_of_nodes = transport.get_number_of_nodes()
        gamma = 1 / ((2 * omega / number_of_nodes + 1) * liptschitz_gradient_constant + 
                     2 * omega / number_of_nodes * bregman_smoothness_variance)
        return gamma


@FactoryNode.register("dcgd_stochastic_node")
class DCGDStochasticNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(DCGDStochasticNodeAlgorithm, self).__init__(function, compressor, **kwargs)
    
    def calculate_message(self, point):
        stochastic_gradient = self._function.stochastic_gradient(point)
        return self._compressor.compress(stochastic_gradient)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryMaster.register("dcgd_stochastic_master")
class DCGDStochasticMasterAlgorithm(BaseTransportMasterAlgorithm):
        def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None, meta=None):
            super(DCGDStochasticMasterAlgorithm, self).__init__(transport, meta)
            self._point = point
            self._gamma = gamma
            if gamma_multiply is not None:
                self._gamma *= gamma_multiply
    
        def step(self):
            self._point = self._point - self._gamma * self.calculate_message()

        def get_point(self):
            return self._point
        
        def calculate_message(self):
            messages = self._transport.call_nodes_method(node_method="calculate_message", 
                                                         point=self._point)
            return self._average_compressed(messages)
        
        def calculate_function(self):
            gradients = self._transport.call_nodes_method(node_method="calculate_function", 
                                                          point=self._point)
            return self._average(gradients)
        
        def calculate_gradient(self):
            gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                          point=self._point)
            return self._average(gradients)


@FactoryNode.register("dcgd_permutation_node")
class DCGDPermutationNodeAlgorithm(RandDianaPermutationNodeAlgorithm):
    pass


@FactoryMaster.register("dcgd_permutation_master")
class DCGDPermutationMasterAlgorithm(RandDianaPermutationMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        gamma = gamma if gamma is not None else self._estimate_gamma(transport, meta)
        super(DCGDPermutationMasterAlgorithm, self).__init__(transport=transport, point=point, gamma=gamma, 
                                                             gamma_multiply=gamma_multiply, 
                                                             prob=0.0, seed=seed,
                                                             meta=meta)
        
    @classmethod
    def _estimate_gamma(cls, transport, meta):
        liptschitz_gradient_constant, _, bregman_smoothness_variance = \
            cls._liptschitz_gradient_constants(transport, meta)
        omega = cls._omega(transport)
        number_of_nodes = transport.get_number_of_nodes()
        gamma = 1 / ((omega + 1) / number_of_nodes * liptschitz_gradient_constant + 
                     ((omega + 1) / number_of_nodes + 1) * bregman_smoothness_variance)
        return gamma


@FactoryNode.register("diana_node")
class DianaNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(DianaNodeAlgorithm, self).__init__(*args, **kwargs)
        assert not self.compressor_biased()
        self._alpha_up = 1 / (self._compressor.omega() + 1)
        
    def _init_local(self, point):
        gradient = self.calculate_gradient(point)
        self._local_gradient_estimator = gradient
        return gradient
    
    def get_alpha_up(self):
        return self._alpha_up
    
    def _calculate_message(self, point):
        gradient = self.calculate_gradient(point)
        compressed_message = self._compressor.compress(gradient - self._local_gradient_estimator)
        self._local_gradient_estimator = self._local_gradient_estimator + self._alpha_up * compressed_message.decompress()
        return compressed_message
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("diana_master")
class DianaMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(DianaMasterAlgorithm, self).__init__(transport, meta)
        self._gamma = gamma
        print("Gamma: {}".format(self._gamma))
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._point = point
        self._init_nodes()
    
    def _init_nodes(self):
        gradients = self._transport.call_nodes_method(
            node_method="_init_local", point=BroadcastNumpy(self._point))
        self._gradient_estimator = self._average(gradients)
        alphas = self._transport.call_nodes_method(node_method="get_alpha_up")
        for lhs, rhs in zip(alphas[1:], alphas[:-1]):
            assert lhs == rhs
        self._alpha_up = alphas[0]
    
    def step(self):
        messages = self._transport.call_nodes_method(node_method="_calculate_message",
                                                     point=self._point)
        message = self._average_compressed(messages)
        gradient_estimator = self._gradient_estimator + message
        self._gradient_estimator = self._gradient_estimator + self._alpha_up * message
        self._point = self._point - self._gamma * gradient_estimator
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(self._point))
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(self._point))
        return self._average(function_values)


class BaseEF21NodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(BaseEF21NodeAlgorithm, self).__init__(*args, **kwargs)
        assert self.compressor_biased()
        self._gradient_estimator = None
        
    def _init(self, point, init_with_gradients):
        gradient = self.calculate_gradient(point)
        if not init_with_gradients:
            compressed_gradient = self._compressor.compress(gradient)
            self._gradient_estimator = compressed_gradient.decompress()
            return compressed_gradient
        else:
            self._gradient_estimator = gradient
            return gradient
    
    def _calculate_message(self, point):
        gradient = self._gradient_estimator_impl(point)
        compressed_message = self._compressor.compress(gradient - self._gradient_estimator)
        message = compressed_message.decompress()
        self._gradient_estimator = self._gradient_estimator + message
        return compressed_message
    
    def _gradient_estimator(self, point):
        raise NotImplementedError()
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def alpha(self):
        return self._compressor.alpha()


@FactoryNode.register("ef21_node")
class EF21NodeAlgorithm(BaseEF21NodeAlgorithm):
    def _gradient_estimator_impl(self, point):
        return self.calculate_gradient(point)


@FactoryNode.register("ef21_polyak_node")
class EF21PolyakNodeAlgorithm(BaseEF21NodeAlgorithm):
    def _gradient_estimator_impl(self, point):
        return self.calculate_gradient(point)


@FactoryNode.register("ef21_stochastic_node")
class EF21StochasticNodeAlgorithm(BaseEF21NodeAlgorithm):
    def __init__(self, *args, momentum=1.0, double_momentum=False, **kwargs):
        super(EF21StochasticNodeAlgorithm, self).__init__(*args, **kwargs)
        # for simplicity, we take an exact gradient
        self._stochastic_gradient_estimator = None
        self._second_stochastic_gradient_estimator = None
        self._momentum = momentum
        self._double_momentum = double_momentum
        
    def _init(self, *args, **kwargs):
        output = super(EF21StochasticNodeAlgorithm, self)._init(*args, **kwargs)
        self._stochastic_gradient_estimator = self._gradient_estimator
        if self._double_momentum:
            self._second_stochastic_gradient_estimator = self._gradient_estimator
        return output

    def _gradient_estimator_impl(self, point):
        stochastic_gradient = self._function.stochastic_gradient(point)
        self._stochastic_gradient_estimator = \
            ((1 - self._momentum) * self._stochastic_gradient_estimator 
             + self._momentum * stochastic_gradient)
        if self._double_momentum:
            self._second_stochastic_gradient_estimator = \
                ((1 - self._momentum) * self._second_stochastic_gradient_estimator 
                 + self._momentum * self._stochastic_gradient_estimator)
            return self._second_stochastic_gradient_estimator
        else:
            return self._stochastic_gradient_estimator


class BaseEF21MasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 init_with_gradients=False,
                 meta=None):
        super(BaseEF21MasterAlgorithm, self).__init__(transport, meta)
        self._gamma = gamma if gamma is not None else self._estimate_gamma()
        print("Gamma: {}".format(self._gamma))
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._point = point
        self._is_torch = False
        self._init(point, init_with_gradients)
    
    def _init(self, point, init_with_gradients):
        self._is_torch = torch.is_tensor(point)
        if not self._is_torch:
            point=BroadcastNumpy(point)
        compressed_gradient_estimators = self._transport.call_nodes_method(
            node_method="_init", point=point, init_with_gradients=init_with_gradients)
        if not init_with_gradients:
            self._gradient_estimator = self._average_compressed(compressed_gradient_estimators)
        else:
            self._gradient_estimator = self._average(compressed_gradient_estimators)

    def _update_gamma(self):
        pass

    def step(self):
        self._update_gamma()
        self._point = self._point - self._gamma * self._gradient_estimator
        point = self._point
        if not self._is_torch:
            point=BroadcastNumpy(point)
        messages = self._transport.call_nodes_method(node_method="_calculate_message", point=point)
        message = self._average_compressed(messages)
        self._gradient_estimator = self._gradient_estimator + message
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        point = self._point
        if not self._is_torch:
            point=BroadcastNumpy(point)
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient", point=point)
        return self._average(gradients)
    
    def calculate_function(self):
        point = self._point
        if not self._is_torch:
            point=BroadcastNumpy(point)
        function_values = self._transport.call_nodes_method(node_method="calculate_function", point=point)
        return self._average(function_values)
    
    @staticmethod
    def _alpha(transport):
        alphas = transport.call_nodes_method(node_method="alpha")
        for l, r in zip(alphas[:-1], alphas[1:]):
            assert l == r
        return alphas[0]


@FactoryMaster.register("ef21_master")
class EF21MasterAlgorithm(BaseEF21MasterAlgorithm):
    @staticmethod
    def _tilde_liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.sqrt(np.mean(np.square(liptschitz_gradient_constants)))
    
    @staticmethod
    def _liptschitz_gradient_constant(transport):
        liptschitz_gradient_constants = transport.call_nodes_method(node_method="liptschitz_gradient_constant")
        return np.mean(liptschitz_gradient_constants)
    
    def _estimate_gamma(self):
        liptschitz_gradient_constant = self._liptschitz_gradient_constant(self._transport)
        tilde_liptschitz_gradient_constant = self._tilde_liptschitz_gradient_constant(self._transport)
        alpha = self._alpha(self._transport)
        sqrt_beta_theta = 2 / alpha - 1
        inv_gamma = liptschitz_gradient_constant + tilde_liptschitz_gradient_constant * sqrt_beta_theta
        gamma = 1. / inv_gamma
        return gamma


@FactoryMaster.register("ef21_polyak_master")
class EF21PolyakMasterAlgorithm(BaseEF21MasterAlgorithm):
    def __init__(self, *args, min_function_value=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert min_function_value is not None
        self._min_function_value = min_function_value
    
    def _update_gamma(self):
        function_value = self.calculate_function()
        norm_gradient_estimator = norm_numpy_or_torch(self._gradient_estimator) ** 2
        self._gamma = (function_value - self._min_function_value) / norm_gradient_estimator


@FactoryMaster.register("ef21_stochastic_master")
class EF21StochasticMasterAlgorithm(BaseEF21MasterAlgorithm):
    pass


class BaseBeerNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(BaseBeerNodeAlgorithm, self).__init__(*args, **kwargs)
        assert self.compressor_biased()
        
    def _init(self, point, gamma):
        self._mix_gamma = self.alpha()
        self._gamma = gamma
        self._x = point
        self._g = 0
        self._h = 0
        self._v = self.calculate_gradient(point)
    
    def _calculate_h(self, mix_h):
        self._prev_x = self._x
        self._x = self._x + self._mix_gamma * (mix_h - self._h) - self._gamma * self._v
        compressed_message = self._compressor.compress(self._x - self._h)
        message = compressed_message.decompress()
        self._h = self._h + message
        return compressed_message
    
    def _calculate_g(self, mix_g):
        self._v = (self._v + self._mix_gamma * (mix_g - self._g) + 
                   self._gradient_estimator_impl(self._x) - self._gradient_estimator_impl(self._prev_x))
        compressed_message = self._compressor.compress(self._v - self._g)
        message = compressed_message.decompress()
        self._g = self._g + message
        return compressed_message
    
    def _gradient_estimator_impl(self, point):
        raise NotImplementedError()
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def get_point(self):
        return self._x
    
    def alpha(self):
        return self._compressor.alpha()


@FactoryNode.register("beer_stochastic_node")
class BeerStochasticNodeAlgorithm(BaseBeerNodeAlgorithm):
    def _gradient_estimator_impl(self, point):
        return self._function.stochastic_gradient(point)


class BaseBeerMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(BaseBeerMasterAlgorithm, self).__init__(transport, meta)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._init(point, gamma)
    
    def _init(self, point, gamma):
        self._transport.call_nodes_method(
            node_method="_init", point=BroadcastNumpy(point), gamma=gamma)
        self._mix_h = 0
        self._mix_g = 0
    
    def step(self):
        messages = self._transport.call_nodes_method(node_method="_calculate_h",
                                                     mix_h=self._mix_h)
        message = self._average_compressed(messages)
        self._mix_h = self._mix_h + message
        messages = self._transport.call_nodes_method(node_method="_calculate_g",
                                                     mix_g=self._mix_g)
        message = self._average_compressed(messages)
        self._mix_g = self._mix_g + message
    
    def get_point(self):
        return self._average(self._transport.call_nodes_method(node_method="get_point"))
    
    def calculate_gradient(self):
        point = self.get_point()
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(point))
        return self._average(gradients)
    
    def calculate_function(self):
        point = self.get_point()
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(point))
        return self._average(function_values)


@FactoryMaster.register("beer_stochastic_master")
class BeerStochasticMasterAlgorithm(BaseBeerMasterAlgorithm):
    pass


# Using the algorithm from the EF = EF21 project
class BaseEFNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(BaseEFNodeAlgorithm, self).__init__(*args, **kwargs)
        assert self.compressor_biased()
        self._point_estimator = None
        self._global_point_estimator = None
        
    def _init(self, point, gamma):
        self._point_estimator = point
        self._global_point_estimator = point
        self._gamma = gamma
    
    def _calculate_message(self, point):
        gradient = self._gradient_estimator_impl(point)
        self._point_estimator = self._point_estimator - self._gamma * gradient
        compressed_message = self._compressor.compress(self._point_estimator - self._global_point_estimator)
        message = compressed_message.decompress()
        self._global_point_estimator = self._global_point_estimator + message
        return compressed_message
    
    def _gradient_estimator(self, point):
        raise NotImplementedError()
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def alpha(self):
        return self._compressor.alpha()


@FactoryNode.register("ef_stochastic_node")
class EFNodeAlgorithm(BaseEFNodeAlgorithm):
    def _gradient_estimator_impl(self, point):
        return self._function.stochastic_gradient(point)


class BaseEFMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(BaseEFMasterAlgorithm, self).__init__(transport, meta)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        self._point = point
        self._is_torch = False
        self._init(point, gamma)
    
    def _init(self, point, gamma):
        self._is_torch = torch.is_tensor(point)
        if not self._is_torch:
            point=BroadcastNumpy(point)
        self._transport.call_nodes_method(node_method="_init", point=point, gamma=gamma)
    
    def step(self):
        point = self._point
        if not self._is_torch:
            point=BroadcastNumpy(point)
        messages = self._transport.call_nodes_method(node_method="_calculate_message", point=point)
        message = self._average_compressed(messages)
        self._point = self._point + message
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        point = self._point
        if not self._is_torch:
            point=BroadcastNumpy(point)
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient", point=point)
        return self._average(gradients)
    
    def calculate_function(self):
        point = self._point
        if not self._is_torch:
            point=BroadcastNumpy(point)
        function_values = self._transport.call_nodes_method(node_method="calculate_function", point=point)
        return self._average(function_values)


@FactoryMaster.register("ef_stochastic_master")
class EFStochasticMasterAlgorithm(BaseEFMasterAlgorithm):
    pass


class BaseNeolithicNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(BaseNeolithicNodeAlgorithm, self).__init__(*args, **kwargs)
        assert self.compressor_biased()
        self._error = 0
        
    def init_node(self, point, gamma):
        self._point = point
        self._gamma = gamma
    
    def calculate_message(self):
        gradient = self._gradient_impl(self._point)
        error_compensate = self._error + gradient
        compressed_message = self._compressor.compress(error_compensate)
        message = compressed_message.decompress()
        self._error = error_compensate - message
        return compressed_message
    
    def _gradient_impl(self):
        raise NotImplementedError()
    
    def step(self, message):
        message = message.decompress()
        self._point = self._point - self._gamma * message
        
    def get_point(self):
        return self._point
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)
    

@FactoryNode.register("neolithic_node")
class NeolithicNodeAlgorithm(BaseNeolithicNodeAlgorithm):
    def _gradient_impl(self, point):
        return self.calculate_gradient(point)


@FactoryNode.register("neolithic_stochastic_node")
class NeolithicStochasticNodeAlgorithm(BaseNeolithicNodeAlgorithm):
    def _gradient_impl(self, point):
        return self._function.stochastic_gradient(point)


class BaseNeolithicMasterAlgorithm(BaseCompressorMasterAlgorithm):
    def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(BaseNeolithicMasterAlgorithm, self).__init__(transport, meta, compressor)
        self._gamma = gamma if gamma is not None else self._estimate_gamma()
        print("Gamma: {}".format(self._gamma))
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._error = 0
        self._transport.call_nodes_method(node_method="init_node",
                                          point=point,
                                          gamma=self._gamma)
    
    def step(self):
        compressed_messages = self._transport.call_nodes_method(node_method="calculate_message")
        messages = [compressed_message.decompress() for compressed_message in compressed_messages]
        average_message = self._average(messages)
        error_compensate = self._error + average_message
        compressed_message = self._compressor.compress(error_compensate)
        message = compressed_message.decompress()
        self._error = error_compensate - message
        self._transport.call_nodes_method(node_method="step",
                                          message=compressed_message)
    
    def get_point(self):
        return self._transport.call_node_method(node_index=0, node_method="get_point")
    
    def calculate_gradient(self):
        point = self.get_point()
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(point))
        return self._average(gradients)
    
    def calculate_function(self):
        point = self.get_point()
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(point))
        return self._average(function_values)


@FactoryMaster.register("neolithic_master")
class NeolithicMasterAlgorithm(BaseNeolithicMasterAlgorithm):
    pass


@FactoryMaster.register("neolithic_stochastic_master")
class NeolithicStochasticMasterAlgorithm(BaseNeolithicMasterAlgorithm):
    pass


class DecompressPointEF21NodeAlgorithm(object):
    def __init__(self, state):
        self._state = state
    
    def decompress(self, message):
        decompressed_message = message.decompress()
        self._state = self._state + decompressed_message
        return self._state


class CompressPointEF21MasterAlgorithm(object):
    def __init__(self, compressor, state):
        self._compressor = compressor
        assert self._compressor.biased()
        self._state = state
    
    def compress(self, point):
        message = self._compressor.compress(point - self._state)
        self._state = self._state + message.decompress()
        return message
    
    def get_state(self):
        return self._state


@FactoryNode.register("dcgd_ef21_primal_node")
class DCGDEF21PrimalNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(DCGDEF21PrimalNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def init_decompressor_manager(self, state):
        self._decompressor_manager = DecompressPointEF21NodeAlgorithm(state)
    
    def calculate_message(self, message):
        point = self._decompressor_manager.decompress(message)
        stochastic_gradient = self._function.gradient(point)
        return self._compressor.compress(stochastic_gradient)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryMaster.register("dcgd_ef21_primal_master")
class DCGDEF21PrimalMasterAlgorithm(BaseCompressorMasterAlgorithm):
        def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None, seed=None, meta=None):
            super(DCGDEF21PrimalMasterAlgorithm, self).__init__(transport, meta, compressor)
            self._point = point
            self._gamma = gamma
            if gamma_multiply is not None:
                self._gamma *= gamma_multiply
            self._compressor_manager = CompressPointEF21MasterAlgorithm(self._compressor, state=point)
            self._transport.call_nodes_method(node_method="init_decompressor_manager", state=point)
    
        def step(self):
            message = self._compressor_manager.compress(self._point)
            messages = self._transport.call_nodes_method(node_method="calculate_message",
                                                         message=message)
            gradient_estimator = self._average_compressed(messages)
            self._point = self._point - self._gamma * gradient_estimator

        def get_point(self):
            return self._point
        
        def calculate_function(self):
            gradients = self._transport.call_nodes_method(node_method="calculate_function", 
                                                          point=self._point)
            return self._average(gradients)
        
        def calculate_gradient(self):
            gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                          point=self._point)
            return self._average(gradients)


@FactoryNode.register("ef21_bc_node")
class EF21BCNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(EF21BCNodeAlgorithm, self).__init__(*args, **kwargs)
        assert self.compressor_biased()
        self._first_iter = True
        
    def _init_local(self, point):
        gradient = self.calculate_gradient(point)
        self._local_gradient_estimator = gradient
        self._point = point
        return gradient
    
    def _init_global(self, gradient_estimator, gamma):
        self._gradient_estimator = gradient_estimator
        self._gamma = gamma
    
    def _calculate_message(self, message):
        self._point = self._point - self._gamma * self._gradient_estimator
        if message is not None:
            self._gradient_estimator = self._gradient_estimator + message.decompress()
        else:
            assert self._first_iter
            self._first_iter = False
        gradient = self.calculate_gradient(self._point)
        compressed_message = self._compressor.compress(gradient - self._local_gradient_estimator)
        self._local_gradient_estimator = self._local_gradient_estimator + compressed_message.decompress()
        return compressed_message
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("ef21_bc_master")
class EF21BCMasterAlgorithm(BaseCompressorMasterAlgorithm):
    def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(EF21BCMasterAlgorithm, self).__init__(transport, meta, compressor)
        assert self._compressor.biased()
        self._gamma = gamma if gamma is not None else self._estimate_gamma()
        print("Gamma: {}".format(self._gamma))
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._point = point
        self._init_nodes()
    
    def _init_nodes(self):
        gradients = self._transport.call_nodes_method(
            node_method="_init_local", point=BroadcastNumpy(self._point))
        self._gradient_estimator = self._average(gradients)
        self._local_gradient_estimator = self._average(gradients)
        self._transport.call_nodes_method(node_method="_init_global",
                                          gradient_estimator=self._gradient_estimator,
                                          gamma=self._gamma)
        self._message_to_nodes = None
    
    def step(self):
        self._point = self._point - self._gamma * self._gradient_estimator
        messages = self._transport.call_nodes_method(node_method="_calculate_message",
                                                     message=self._message_to_nodes)
        message = self._average_compressed(messages)
        self._local_gradient_estimator = self._local_gradient_estimator + message
        self._message_to_nodes = self._compressor.compress(
            self._local_gradient_estimator - self._gradient_estimator)
        self._gradient_estimator = self._gradient_estimator + self._message_to_nodes.decompress()
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(self._point))
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(self._point))
        return self._average(function_values)


@FactoryNode.register("mcm_node")
class MCMNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(MCMNodeAlgorithm, self).__init__(*args, **kwargs)
        assert not self.compressor_biased()
        self._first_iter = True
        self._alpha_up = 1 / (2 * (self._compressor.omega() + 1))
        
    def _init_local(self, point, alpha_down):
        gradient = self.calculate_gradient(point)
        self._local_gradient_estimator = gradient
        self._point_estimator = point
        self._alpha_down = alpha_down
        return gradient
    
    def get_alpha_up(self):
        return self._alpha_up
    
    def _calculate_message(self, message):
        if message is not None:
            decompressed_message = message.decompress()
            point = self._point_estimator + decompressed_message
            self._point_estimator = self._point_estimator + self._alpha_down * decompressed_message
        else:
            assert self._first_iter
            self._first_iter = False
            point = self._point_estimator
        gradient = self.calculate_gradient(point)
        compressed_message = self._compressor.compress(gradient - self._local_gradient_estimator)
        self._local_gradient_estimator = self._local_gradient_estimator + self._alpha_up * compressed_message.decompress()
        return compressed_message
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("mcm_master")
class MCMMasterAlgorithm(BaseCompressorMasterAlgorithm):
    def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(MCMMasterAlgorithm, self).__init__(transport, meta, compressor)
        assert not self._compressor.biased()
        self._gamma = gamma if gamma is not None else self._estimate_gamma()
        print("Gamma: {}".format(self._gamma))
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._point = point
        self._point_estimator = point
        self._alpha_down = 1 / (2 * (self._compressor.omega() + 1))
        self._init_nodes()
    
    def _init_nodes(self):
        gradients = self._transport.call_nodes_method(
            node_method="_init_local", point=BroadcastNumpy(self._point),
            alpha_down=self._alpha_down)
        self._gradient_estimator = self._average(gradients)
        self._message_to_nodes = None
        alphas = self._transport.call_nodes_method(node_method="get_alpha_up")
        for lhs, rhs in zip(alphas[1:], alphas[:-1]):
            assert lhs == rhs
        self._alpha_up = alphas[0]
    
    def step(self):
        messages = self._transport.call_nodes_method(node_method="_calculate_message",
                                                     message=self._message_to_nodes)
        message = self._average_compressed(messages)
        gradient_estimator = self._gradient_estimator + message
        self._gradient_estimator = self._gradient_estimator + self._alpha_up * message
        self._point = self._point - self._gamma * gradient_estimator
        self._message_to_nodes = self._compressor.compress(
            self._point - self._point_estimator)
        self._point_estimator = self._point_estimator + self._alpha_down * self._message_to_nodes.decompress()
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(self._point))
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(self._point))
        return self._average(function_values)


@FactoryNode.register("diana_ef21_primal_node")
class DianaEF21PrimalNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(DianaEF21PrimalNodeAlgorithm, self).__init__(*args, **kwargs)
        assert not self.compressor_biased()
        self._first_iter = True
        self._alpha_up = 1 / (self._compressor.omega() + 1)
        
    def _init_local(self, point):
        gradient = self.calculate_gradient(point)
        self._local_gradient_estimator = gradient
        self._decompressor_manager = DecompressPointEF21NodeAlgorithm(point)
        self._first_point = point
        return gradient
    
    def get_alpha_up(self):
        return self._alpha_up
    
    def _calculate_message(self, message):
        if message is not None:
            point = self._decompressor_manager.decompress(message)
        else:
            assert self._first_iter
            self._first_iter = False
            point = self._first_point
        gradient = self.calculate_gradient(point)
        compressed_message = self._compressor.compress(gradient - self._local_gradient_estimator)
        self._local_gradient_estimator = self._local_gradient_estimator + self._alpha_up * compressed_message.decompress()
        return compressed_message
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("diana_ef21_primal_master")
class DianaEF21PrimalMasterAlgorithm(BaseCompressorMasterAlgorithm):
    def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None, seed=None,
                 meta=None):
        super(DianaEF21PrimalMasterAlgorithm, self).__init__(transport, meta, compressor)
        assert self._compressor.biased()
        self._gamma = gamma
        print("Gamma: {}".format(self._gamma))
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._point = point
        self._compressor_manager = CompressPointEF21MasterAlgorithm(self._compressor, state=point)
        self._init_nodes()
    
    def _init_nodes(self):
        gradients = self._transport.call_nodes_method(
            node_method="_init_local", point=BroadcastNumpy(self._point))
        self._gradient_estimator = self._average(gradients)
        self._message_to_nodes = None
        alphas = self._transport.call_nodes_method(node_method="get_alpha_up")
        for lhs, rhs in zip(alphas[1:], alphas[:-1]):
            assert lhs == rhs
        self._alpha_up = alphas[0]
    
    def step(self):
        messages = self._transport.call_nodes_method(node_method="_calculate_message",
                                                     message=self._message_to_nodes)
        message = self._average_compressed(messages)
        gradient_estimator = self._gradient_estimator + message
        self._gradient_estimator = self._gradient_estimator + self._alpha_up * message
        self._point = self._point - self._gamma * gradient_estimator
        self._message_to_nodes = self._compressor_manager.compress(self._point)
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(self._point))
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(self._point))
        return self._average(function_values)


def _calculate_learning_rates_for_accelerated_bidirectional_diana(sum_gamma, bar_lipschitz, mu, p, alpha, tau, beta):
    a, b, c = (p * bar_lipschitz * sum_gamma, p * (bar_lipschitz + sum_gamma * mu), -(bar_lipschitz + sum_gamma * mu))
    theta_bar = (2 * c) / (-b - np.sqrt(b**2 - 4 * a * c))
    theta = min(theta_bar, 1 / 4. * np.min([1., alpha / p, tau / p, beta / p]))
    gamma = p * theta * sum_gamma / (1 - p * theta)
    next_sum_gamma = sum_gamma + gamma
    return next_sum_gamma, gamma, theta


@FactoryNode.register("accelerated_bidirectional_diana_node")
class AcceleratedBidirectionalDianaNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, *args, **kwargs):
        super(AcceleratedBidirectionalDianaNodeAlgorithm, self).__init__(*args, **kwargs)
        assert not self.compressor_biased()
        self._beta = 1 / (self._compressor.omega() + 1)
        
    def get_omega(self):
        return self._compressor.omega()
        
    def _init_local(self, point, bar_lipschitz, strongly_convex_constant, sum_gamma):
        self._bar_lipschitz = bar_lipschitz 
        self._strongly_convex_constant = strongly_convex_constant
        self._sum_gamma = sum_gamma
        self._z, self._w, self._y = point, point, point
        self._h = self.calculate_gradient(point)
        return self._h
    
    def _init_local_with_grad(self, gradient):
        self._k = gradient
    
    def _calculate_message_at_point_y(self, theta):
        self._y = theta * self._w + (1 - theta) * self._z
        gradient = self.calculate_gradient(self._y)
        message = self._compressor.compress(gradient - self._h)
        return message
    
    def _calculate_message_at_point_z(self, 
                                      primal_compressed_message, 
                                      primal_noncompressed_message,
                                      dual_noncompressed_message,
                                      gamma, next_sum_gamma):
        q = AcceleratedBidirectionalDianaMasterAlgorithm._gradient_step(
            self._k, self._w, gamma, next_sum_gamma,
            self._sum_gamma, self._bar_lipschitz, self._strongly_convex_constant, self._y)
        if primal_noncompressed_message is not None:
            assert dual_noncompressed_message is not None
            self._z = primal_noncompressed_message
            self._k = dual_noncompressed_message
        self._w = q + primal_compressed_message.decompress()
        gradient = self.calculate_gradient(self._z)
        message = self._compressor.compress(gradient - self._h)
        self._h = self._h + self._beta * message.decompress()
        self._sum_gamma = next_sum_gamma
        return message
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_function(self, point):
        return self._function.value(point)


@FactoryMaster.register("accelerated_bidirectional_diana_master")
class AcceleratedBidirectionalDianaMasterAlgorithm(BaseCompressorMasterAlgorithm):
    def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None, seed=None,
                 meta=None, sum_gamma_inverse_omega=False, add_strongly_convex_constant=0.0,
                 scale_prob=1.0):
        super(AcceleratedBidirectionalDianaMasterAlgorithm, self).__init__(transport, meta, compressor)
        self._sum_gamma_inverse_omega = sum_gamma_inverse_omega
        self._generator = np.random.default_rng(seed)
        assert self._compressor.biased()
        self._gamma = gamma
        if gamma_multiply is not None:
            self._gamma *= gamma_multiply
        self._bar_lipschitz = 1. / self._gamma
        print("Gamma: {}".format(self._gamma))
        self._x, self._u, self._w, self._z = point, point, point, point
        self._compressor_manager = None
        self._strongly_convex_constant = (self._meta.strongly_convex_constant()
                                          + add_strongly_convex_constant)
        # pessimistic choice of parameters
        self._alpha = compressor.alpha()
        self._init_nodes()
        self._beta = 1. / (self._omega + 1)
        if self._omega == 0:
            self._prob = 1.
        else:
            self._prob = max(1. / (self._omega + 1), 
                             np.sqrt(1. / (self._alpha * self._omega * (self._omega + 1)**2)))
        self._prob = self._prob * scale_prob 
        self._prob = min(1., self._prob)
        assert self._prob >= 0.
        if self._omega == 0:
            self._tau = 1.
        else:
            self._tau = min(1., 1. / (self._omega ** (1 / 3.) * (self._omega + 1) ** (2 / 3.)))
    
    def _init_nodes(self):
        omegas = self._transport.call_nodes_method(node_method="get_omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        self._omega = omegas[0]
        if self._sum_gamma_inverse_omega:
            self._sum_gamma = self._omega + 1
        else:
            self._sum_gamma = 1.
        gradients = self._transport.call_nodes_method(
            node_method="_init_local", point=BroadcastNumpy(self._x),
            bar_lipschitz=self._bar_lipschitz, 
            strongly_convex_constant=self._strongly_convex_constant,
            sum_gamma=self._sum_gamma)
        self._h = self._average(gradients)
        self._v = self._h
        self._k = self._h
        self._transport.call_nodes_method(
            node_method="_init_local_with_grad", gradient=self._h)
    
    def step(self):
        next_sum_gamma, gamma, theta = \
            _calculate_learning_rates_for_accelerated_bidirectional_diana(
                self._sum_gamma, self._bar_lipschitz, self._strongly_convex_constant, 
                self._prob, self._alpha, self._tau, self._beta)
        y = theta * self._w + (1 - theta) * self._z
        messages = self._transport.call_nodes_method(node_method="_calculate_message_at_point_y",
                                                     theta=theta)
        message = self._average_compressed(messages)
        gradient_estimator = self._h + message
        self._u = self._gradient_step(gradient_estimator, self._u, gamma, next_sum_gamma,
                                      self._sum_gamma, self._bar_lipschitz, self._strongly_convex_constant, y)
        q = self._gradient_step(self._k, self._w, gamma, next_sum_gamma,
                                self._sum_gamma, self._bar_lipschitz, self._strongly_convex_constant, y)
        primal_compressed_message = self._compressor.compress(self._u - q)
        self._w = q + primal_compressed_message.decompress()
        self._x = theta * self._u + (1 - theta) * self._z
        coin = bernoulli_sample(self._generator, self._prob)
        primal_noncompressed_message = None
        dual_noncompressed_message = None
        if coin:
            self._k = self._v
            self._z = self._x
            primal_noncompressed_message = self._z
            dual_noncompressed_message = self._k
        messages = self._transport.call_nodes_method(node_method="_calculate_message_at_point_z",
                                                     primal_compressed_message=primal_compressed_message,
                                                     primal_noncompressed_message=primal_noncompressed_message,
                                                     dual_noncompressed_message=dual_noncompressed_message,
                                                     gamma=gamma, next_sum_gamma=next_sum_gamma)
        message = self._average_compressed(messages)
        self._v = (1 - self._tau) * self._v + self._tau * (self._h + message)
        self._h = self._h + self._beta * message
        self._sum_gamma = next_sum_gamma
    
    @staticmethod
    def _gradient_step(dual_vector, previous_point, next_gamma, next_sum_gamma,
                       sum_gamma, bar_lipschitz, strongly_convex_constant, y):
        point =  ((bar_lipschitz + sum_gamma * strongly_convex_constant) * previous_point
                  + strongly_convex_constant * next_gamma * y
                  - next_gamma * dual_vector)
        point = point / (bar_lipschitz + next_sum_gamma * strongly_convex_constant)
        return point
    
    def get_point(self):
        return self._x
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=BroadcastNumpy(self._x))
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=BroadcastNumpy(self._x))
        return self._average(function_values)


@FactoryNode.register("zero_marina_ef21_primal_node")
class ZeroMarinaEF21PrimalNodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(ZeroMarinaEF21PrimalNodeAlgorithm, self).__init__(function, compressor, **kwargs)
        
    def init_node(self, point, momentum):
        self._momentum = momentum
        self._decompressor_manager = DecompressPointEF21NodeAlgorithm(point)
        self._gradient_estimator = self._function.gradient(point)
        self._previous_point = point
        return self._gradient_estimator
        
    def step(self, message):
        point = self._decompressor_manager.decompress(message)
        local_gradient_estimator, previous_local_gradient_estimator = \
            self._function.gradient(point),  self._function.gradient(self._previous_point)
        self._previous_point = point
        message = local_gradient_estimator - previous_local_gradient_estimator - \
            self._momentum * (self._gradient_estimator - previous_local_gradient_estimator)
        compressed_message = self._compressor.compress(message)
        decompressed_message = compressed_message.decompress()
        self._gradient_estimator = self._gradient_estimator + decompressed_message
        return compressed_message
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_gradient(self, point):
        return self._function.gradient(point)


@FactoryMaster.register("zero_marina_ef21_primal_master")
class ZeroMarinaEF21PrimalMasterAlgorithm(BaseCompressorMasterAlgorithm):
    def __init__(self, transport, point, compressor, gamma=None, gamma_multiply=None,
                 momentum=None, seed=None, init_with_gradients=True,
                 meta=None):
        super(ZeroMarinaEF21PrimalMasterAlgorithm, self).__init__(transport, meta, compressor)
        momentum = momentum if momentum is not None else self._estimate_momentum()
        assert gamma is not None
        self._momentum = momentum
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._point = point
        self._gamma = gamma
        self._compressor_manager = CompressPointEF21MasterAlgorithm(self._compressor, state=point)
        self._init()
    
    def _init(self):
        initial_gradient_estimator = self._transport.call_nodes_method(
            node_method="init_node", point=self._point, momentum=self._momentum)
        self._gradient_estimator = self._average(initial_gradient_estimator)
    
    def step(self):
        self._point = self._point - self._gamma * self._gradient_estimator
        message = self._compressor_manager.compress(self._point)
        compressed_messages = self._transport.call_nodes_method(node_method="step", 
                                                                message=message)
        average_updates = self._average_compressed(compressed_messages)
        self._gradient_estimator = self._gradient_estimator + average_updates

    def _estimate_momentum(self):
        return 1.0 / (2 * self._omega(self._transport) + 1)
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]
    
    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient",
                                                      point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function",
                                                            point=self._point)
        return self._average(function_values)


@FactoryMaster.register("m3_master")
class M3MasterAlgorithm(BaseCompressorsMasterAlgorithm):
    def __init__(self, transport, point, compressors, gamma=None, 
                 prob=None, prob_primal=None, 
                 prob_scale=None, prob_scale_primal=None, use_vector_compression=False,
                 gamma_multiply=None, seed=None, meta=None):
        super(M3MasterAlgorithm, self).__init__(transport, meta, compressors)
        assert isinstance(compressors[0], PermutationCompressor)
        seeds = [compressor.get_seed() for compressor in compressors]
        assert np.all(l == r for l,r in zip(seeds[1:], seeds[:-1]))
        
        prob = prob if prob is not None else self._estimate_prob(transport)
        prob_primal = prob_primal if prob_primal is not None else self._estimate_prob_primal(transport)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._number_of_nodes = self._transport.get_number_of_nodes()
        self._use_vector_compression = use_vector_compression
            
        omega = self._omega(self._transport)
        omega_primal = self._compressors[0].omega()
        if omega == 0 or omega_primal == 0:
            beta = 1.
        else:
            beta = min(1, np.cbrt(self._number_of_nodes / (omega * (omega + 1) * omega_primal)))

        self._gamma = gamma
        self._prob = prob
        self._prob_primal = prob_primal
        if prob_scale is not None:
            self._prob *= prob_scale
        if prob_scale_primal is not None:
            self._prob_primal *= prob_scale_primal
        self._generator = np.random.default_rng(seed)
        self._gradient_estimator = None
        self._init(point, beta, self._use_vector_compression)
    
    def _init(self, point, beta, use_vector_compression):
        self._point = point
        self._transport.call_nodes_method(node_method="set_point", point=point)
        self._transport.call_nodes_method(node_method="set_beta", beta=beta)
        self._transport.call_nodes_method(node_method="set_use_vector_compression", 
                                          use_vector_compression=use_vector_compression)
        self._update_gradient_estimator_with_checkpoint()
    
    def step(self):
        previous_point = self._point
        self._point = self._point - self._gamma * self._gradient_estimator
        flag_send_point = bool(bernoulli_sample(self._generator, self._prob_primal))
        for node_index in range(self._number_of_nodes):
            if flag_send_point:
                if not self._use_vector_compression:
                    point = self._point
                else:
                    point = GZipVector(self._point)
                self._transport.call_node_method(node_index, node_method="update_primal", 
                                                 flag_send_point=flag_send_point, 
                                                 update=point)
            else:
                compressed_info = self._compressors[node_index].compress(self._point - previous_point)
                self._transport.call_node_method(node_index, node_method="update_primal", 
                                                 flag_send_point=flag_send_point,
                                                 update=compressed_info)
        flag_calculate_gradient = bernoulli_sample(self._generator, self._prob)
        if flag_calculate_gradient:
            self._update_gradient_estimator_with_checkpoint()
        else:
            self._update_gradient_estimator()

    def get_point(self):
        return self._point
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient", point=self._point)
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function", point=self._point)
        return self._average(function_values)
    
    def _update_gradient_estimator_with_checkpoint(self, *args, **kwargs):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient_at_z")
        if self._use_vector_compression:
            gradients = [grad.decompress() for grad in gradients]
        self._gradient_estimator = self._average(gradients)
    
    def _update_gradient_estimator(self):
        updates = self._transport.call_nodes_method(node_method="calculate_updates")
        average_updates = self._average_compressed(updates)
        self._gradient_estimator = self._gradient_estimator + average_updates
    
    @staticmethod
    def _estimate_prob(transport):
        num_nonzero_components = transport.call_nodes_method(node_method="num_nonzero_components")
        dim = transport.call_node_method(node_index=0, node_method="dim")
        return np.mean(num_nonzero_components) / dim
    
    def _estimate_prob_primal(self, transport):
        num_nonzero_components = self._compressors[0].num_nonzero_components()
        dim = transport.call_node_method(node_index=0, node_method="dim")
        return np.mean(num_nonzero_components) / dim
    
    @staticmethod
    def _omega(transport):
        omegas = transport.call_nodes_method(node_method="omega")
        for lhs, rhs in zip(omegas[1:], omegas[:-1]):
            assert lhs == rhs
        return omegas[0]


@FactoryNode.register("m3_node")
class M3NodeAlgorithm(BaseCompressorNodeAlgorithm):
    def __init__(self, function, compressor, **kwargs):
        super(M3NodeAlgorithm, self).__init__(function, compressor, **kwargs)
    
    def set_point(self, point):
        self._w = point
        self._z = point
        
    def set_beta(self, beta):
        self._beta = beta
        
    def set_use_vector_compression(self, use_vector_compression):
        self._use_vector_compression = use_vector_compression
    
    def update_primal(self, flag_send_point, update):
        if flag_send_point:
            if self._use_vector_compression:
                update = update.decompress()
            self._w = update
        else:
            self._w = self._w + update.decompress()
        self._previous_z = self._z
        self._z = self._beta * self._w + (1 - self._beta) * self._z
        
    def calculate_gradient(self, point):
        return self._function.gradient(point)
    
    def calculate_gradient_at_z(self):
        grad = self._function.gradient(self._z)
        if not self._use_vector_compression:
            return grad
        else:
            return GZipVector(grad)
    
    def calculate_function(self, point):
        return self._function.value(point)
    
    def calculate_updates(self):
        update_gradient_estimator = \
            self._compressor.compress(self._function.gradient(self._z) - self._function.gradient(self._previous_z))
        return update_gradient_estimator


@FactoryMaster.register("core_master")
class CoreMasterAlgorithm(BaseTransportMasterAlgorithm):
    def __init__(self, transport, point, gamma=None, 
                 gamma_multiply=None, seed=None, meta=None):
        super(CoreMasterAlgorithm, self).__init__(transport, meta)
        if gamma_multiply is not None:
            gamma *= gamma_multiply
        self._gamma = gamma
        seed = _generate_seed(seed)
        number_of_nodes = self._transport.get_number_of_nodes()
        self._init(point, gamma, seed, number_of_nodes)
    
    def _init(self, point, gamma, seed, number_of_nodes):
        self._transport.call_nodes_method(node_method="set_point", point=point)
        self._transport.call_nodes_method(node_method="set_gamma", gamma=gamma)
        self._transport.call_nodes_method(node_method="set_seed", seed=seed)
        self._transport.call_nodes_method(node_method="set_number_of_nodes", number_of_nodes=number_of_nodes)
    
    def step(self):
        compressed_vectors = self._transport.call_nodes_method(node_method="sparsify")
        compressed_vector = self._average(compressed_vectors, normalize=False)
        compressed_vectors = self._transport.call_nodes_method(node_method="step", compressed_vector=compressed_vector)

    def get_point(self):
        return self._transport.call_node_method(node_index=0, node_method="get_point")
    
    def calculate_gradient(self):
        gradients = self._transport.call_nodes_method(node_method="calculate_gradient")
        return self._average(gradients)
    
    def calculate_function(self):
        function_values = self._transport.call_nodes_method(node_method="calculate_function")
        return self._average(function_values)


@FactoryNode.register("core_node")
class CoreNodeAlgorithm(BaseNodeAlgorithm):
    def __init__(self, function, number_of_coordinates, **kwargs):
        super(CoreNodeAlgorithm, self).__init__(function, **kwargs)
        self._number_of_coordinates = number_of_coordinates
    
    def set_point(self, point):
        self._point = point

    def get_point(self):
        return self._point

    def set_gamma(self, gamma):
        self._gamma = gamma
        
    def set_number_of_nodes(self, number_of_nodes):
        self._number_of_nodes = number_of_nodes
        
    def set_seed(self, seed):
        self._seed = seed
        self._generator = np.random.default_rng(self._seed)
    
    def sparsify(self):
        dim = len(self._point)
        self._gauss_vectors = self._generator.normal(size=(self._number_of_coordinates, dim)).astype(np.float32)
        compressed_vector = self._gauss_vectors @ self._function.gradient(self._point)
        return compressed_vector

    def step(self, compressed_vector):
        gradient_estimator = (self._gauss_vectors.T @ compressed_vector) / float(self._number_of_coordinates * self._number_of_nodes)
        self._point = self._point - self._gamma * gradient_estimator

    def calculate_gradient(self):
        return self._function.gradient(self._point)
    
    def calculate_function(self):
        return self._function.value(self._point)


def _generate_seed(generator):
    return generator.integers(10e9)


def get_algorithm(functions, point, seed, 
                  algorithm_name, algorithm_master_params={}, algorithm_node_params={},
                  compressor_name=None, compressor_params={}, 
                  compressor_master_name=None, compressor_master_params={}, 
                  multiple_master_compressors=False,
                  parallel=False,
                  number_of_processes=1, shared_memory_size=0, shared_memory_len=1,
                  meta=OptimizationProblemMeta()):
    node_name = algorithm_name + "_node"
    master_name = algorithm_name + "_master"
    node_cls = FactoryNode.get(node_name)
    master_cls = FactoryMaster.get(master_name)
    generator = np.random.default_rng(seed)
    if compressor_name is not None:
        compressors = get_compressor_signatures(compressor_name, compressor_params, len(functions), seed)
        nodes = [Signature(node_cls, function, compressor, seed=_generate_seed(generator), **algorithm_node_params) 
                 for function, compressor in zip(functions, compressors)]
    else:
        nodes = [Signature(node_cls, function, seed=_generate_seed(generator), **algorithm_node_params) 
                 for function in functions]
    transport = Transport(nodes, parallel=parallel, shared_memory_size=shared_memory_size,
                          shared_memory_len=shared_memory_len,
                          number_of_processes=number_of_processes)
    if compressor_master_name is not None:
        number_of_master_compressors = 1 if not multiple_master_compressors else len(functions)
        compressor_master = get_compressors(compressor_master_name, compressor_master_params, 
                                            total_number_of_nodes=number_of_master_compressors, seed=seed)
        if not multiple_master_compressors:
            compressor_master = compressor_master[0]
        return master_cls(transport, point, compressor_master, 
                          seed=seed, meta=meta, **algorithm_master_params)
    else:
        return master_cls(transport, point, seed=seed, meta=meta, **algorithm_master_params)
