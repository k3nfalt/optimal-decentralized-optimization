import copy
import gzip 
from io import BytesIO
import sys

import numpy as np
import torch

from distributed_optimization_library.factory import Factory
from distributed_optimization_library.signature import Signature


NUMBER_OF_BITES_IN_BYTE = 8
NUMBER_OF_BITES_IN_FLOAT32 = 32


def _generate_seed(generator):
    return generator.integers(10e9)


class FactoryCompressor(Factory):
    pass

class BaseCompressor(object):
    BIASED = None
    INDEPENDENT = None
    RANDOM = None
    
    def compress(self, vector):
        raise NotImplementedError()

    def num_nonzero_components(self):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()
    
    @classmethod
    def biased(cls):
        assert cls.BIASED is not None
        return cls.BIASED
    
    @classmethod
    def independent(cls):
        assert cls.INDEPENDENT is not None
        return cls.INDEPENDENT
    
    @classmethod
    def random(cls):
        assert cls.RANDOM is not None
        return cls.RANDOM
    
    @classmethod
    def get_compressor_signatures(cls, params, total_number_of_nodes, seed):
        assert cls.independent()
        generator = np.random.default_rng(seed)
        signatures = []
        for _ in range(total_number_of_nodes):
            if cls.random():
                unique_seed = _generate_seed(generator)
                signatures.append(Signature(cls, seed=unique_seed, **params))
            else:
                signatures.append(Signature(cls, **params))
        return signatures


class SameSeedCompressor(BaseCompressor):
    @classmethod
    def get_compressor_signatures(cls, params, total_number_of_nodes, seed):
        assert not cls.independent()
        assert cls.random()
        
        generator = np.random.default_rng(seed)
        seed = _generate_seed(generator)
        assert params['total_number_of_nodes'] == total_number_of_nodes
    
        signatures = []
        for node_index in range(total_number_of_nodes):
            signatures.append(Signature(cls, node_index, seed=seed, **params))
        return signatures


class UnbiasedBaseCompressor(BaseCompressor):
    BIASED = False
    def omega(self):
        raise NotImplementedError()


class BiasedBaseCompressor(BaseCompressor):
    BIASED = True
    def alpha(self):
        raise NotImplementedError()


class CompressedVector(object):
    def __init__(self, indices, values, dim):
        assert isinstance(values, np.ndarray) and values.ndim == 1
        assert len(indices) == len(values)
        self._indices = indices
        self._values = values
        self._dim = dim
        
    def get_raw_info(self):
        return self._indices, self._values, self._dim
        
    def decompress(self):
        decompressed_array = np.zeros((self._dim,), dtype=self._values.dtype)
        decompressed_array[self._indices] = self._values
        return decompressed_array
        
    def size_in_memory(self):
        #  Omitted self._indices
        return len(self._values) * self._values.itemsize * NUMBER_OF_BITES_IN_BYTE
    
    def number_of_elements(self):
        return len(self._values)


class SumCompressedVectors(object):
    def __init__(self, compressed_vectors):
        assert isinstance(compressed_vectors, (list, tuple))
        assert len(compressed_vectors) > 0
        self._compressed_vectors = compressed_vectors
        
    def decompress(self):
        return sum([compressed_vector.decompress() for compressed_vector in self._compressed_vectors])
        
    def size_in_memory(self):
        return sum([compressed_vector.size_in_memory() for compressed_vector in self._compressed_vectors])


class CompressedTorchVector(object):
    def __init__(self, indices, values, dim):
        assert torch.is_tensor(indices)
        assert torch.is_tensor(values) and values.ndim == 1 and \
            values.dtype == torch.float32
        
        assert len(indices) == len(values)
        self._indices = indices
        self._values = values
        self._dim = dim
        
    def decompress(self):
        decompressed_array = torch.zeros((self._dim,), dtype=self._values.dtype,
                                         device=self._values.device)
        decompressed_array[self._indices] = self._values
        return decompressed_array
        
    def size_in_memory(self):
        #  Omitted self._indices
        return len(self._values) * NUMBER_OF_BITES_IN_FLOAT32


class GZipVector(object):
    def __init__(self, vector):
        assert isinstance(vector, np.ndarray)
        out = BytesIO()
        with gzip.GzipFile(fileobj=out, mode="w") as f:
            np.save(file=f, arr=vector)
        out.seek(0)
        self._size_in_memory = out.getbuffer().nbytes
        with gzip.GzipFile(fileobj=out, mode="r") as f:
            vector_restore = np.load(f)
        self._vector = vector_restore
        assert np.array_equal(vector, vector_restore)
        
    def decompress(self):
        return self._vector
        
    def size_in_memory(self):
        return self._size_in_memory


class BasePermutationCompressor(UnbiasedBaseCompressor, SameSeedCompressor):
    INDEPENDENT = False
    RANDOM = True
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        assert node_number < total_number_of_nodes
        self._node_number = node_number
        self._total_number_of_nodes = total_number_of_nodes
        self._dim = dim
        self._seed = seed
        self._generator = np.random.default_rng(seed=seed)

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        nodes_random_permutation = self._generator.permutation(
            self._total_number_of_nodes)
        block_number = nodes_random_permutation[self._node_number]
        start_dim, end_dim = self._end_start_dim(self._dim, block_number)
        random_permutation = self._get_coordiante_permutation(dim)
        correction_bias = self._total_number_of_nodes
        indices = random_permutation[start_dim:end_dim]
        values = vector[indices] * correction_bias
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector
    
    def _get_coordiante_permutation(self, dim):
        raise NotImplementedError()
    
    def num_nonzero_components(self):
        assert self._dim is not None
        return self._dim / float(self._total_number_of_nodes)
    
    def omega(self):
        assert self._dim is not None
        return self._total_number_of_nodes - 1
    
    def get_seed(self):
        return self._seed
    
    def _end_start_dim(self, dim, block_number):
        assert dim >= self._total_number_of_nodes
        block_size = dim // self._total_number_of_nodes
        residual = dim - block_size * self._total_number_of_nodes
        if block_number < residual:
            start_dim = block_number * (block_size + 1)
            end_dim = min(start_dim + block_size + 1, dim)
        else:
            start_dim = residual * (block_size + 1) + (block_number - residual) * block_size
            end_dim = min(start_dim + block_size, dim)
        return start_dim, end_dim


@FactoryCompressor.register("permutation")
class PermutationCompressor(BasePermutationCompressor):
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        super(PermutationCompressor, self).__init__(
            node_number, total_number_of_nodes, seed, dim)
    
    def _get_coordiante_permutation(self, dim):
        return self._generator.permutation(dim)


class CompressedNatVector(object):
    _EXPONENT_PLUS_SIGN_SIZE = 9
    def __init__(self, indices, values, dim):
        assert isinstance(values, np.ndarray) and values.ndim == 1
        assert len(indices) == len(values)
        self._indices = indices
        self._values = values
        self._dim = dim
        
    def decompress(self):
        decompressed_array = np.zeros((self._dim,), dtype=self._values.dtype)
        decompressed_array[self._indices] = self._values
        return decompressed_array
        
    def size_in_memory(self):
        #  Omitted self._indices. Can be restored using the same random generator.
        return len(self._values) * self._EXPONENT_PLUS_SIGN_SIZE


def _natural_compressor(values):
    assert values.dtype == np.float32
    sign = np.sign(values)
    alpha = np.log2(np.abs(values))
    alpha_down = np.floor(alpha)
    alpha_up = np.ceil(alpha)
    pt = (np.power(2, alpha_up) - np.abs(values)) / np.power(2, alpha_down)
    # TODO: fix this
    random_uniform = np.random.rand(len(values))
    down = random_uniform < pt
    out = np.zeros_like(values)
    out[down] = (sign * np.power(2, alpha_down))[down]
    out[~down] = (sign * np.power(2, alpha_up))[~down]
    out[values == 0.0] = 0.0
    return out


# TODO: omega slightly different
@FactoryCompressor.register("_permutation_with_nat")
class PermutationWithNatCompressor(PermutationCompressor):
    def compress(self, vector):
        compressed_vector = super().compress(vector)
        indices, values, dim = compressed_vector.get_raw_info()
        out = _natural_compressor(values)
        return CompressedNatVector(indices, out, dim)


@FactoryCompressor.register("permutation_fixed_blocks")
class PermutationFixedBlocksCompressor(BasePermutationCompressor):
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        super(PermutationFixedBlocksCompressor, self).__init__(
            node_number, total_number_of_nodes, seed, dim)
    
    def _get_coordiante_permutation(self, dim):
        return np.arange(dim)


@FactoryCompressor.register("group_permutation")
class GroupPermutationCompressor(PermutationCompressor):
    @classmethod
    def get_compressor_signatures(cls, params, total_number_of_nodes, seed):
        nodes_indices_splits = params['nodes_indices_splits']
        assert nodes_indices_splits[0] == 0
        assert nodes_indices_splits[-1] == total_number_of_nodes
        generator = np.random.default_rng(seed=seed)
        signatures = []
        for start, end in zip(nodes_indices_splits[:-1], nodes_indices_splits[1:]):
            total_number_of_nodes_group = end - start
            seed = _generate_seed(generator)
            for node_index in range(total_number_of_nodes_group):
                signatures.append(Signature(cls, node_index, 
                                            total_number_of_nodes=total_number_of_nodes_group, 
                                            seed=seed,
                                            dim=params.get('dim', None)))
        return signatures


@FactoryCompressor.register("nodes_permutation")
class NodesPermutationCompressor(UnbiasedBaseCompressor, SameSeedCompressor):
    INDEPENDENT = False
    RANDOM = True
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        self._node_number = node_number
        self._total_number_of_nodes = total_number_of_nodes
        self._dim = dim
        self._generator = np.random.default_rng(seed=seed)
        self._indices = None

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._dim <= self._total_number_of_nodes and \
            self._total_number_of_nodes % self._dim == 0
        num_coordinates_per_node = self._total_number_of_nodes / self._dim
        if self._indices is None:
            self._indices = np.arange(self._dim)
            self._indices = np.repeat(self._indices, num_coordinates_per_node)
        random_permutation = self._generator.permutation(self._indices)
        indices = random_permutation[self._node_number:self._node_number + 1]
        values = vector[indices] * self._dim
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector

    def num_nonzero_components(self):
        return 1
    
    def omega(self):
        return self._dim - 1


@FactoryCompressor.register("identity_unbiased")
class IdentityUnbiasedCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def __init__(self, dim=None):
        self._dim = dim
    
    def compress(self, vector):
        dim = vector.shape[0]
        compressed_vector = CompressedVector(np.arange(dim), np.copy(vector), dim)
        return compressed_vector
    
    def omega(self):
        return 0
    
    def num_nonzero_components(self):
        return self._dim


@FactoryCompressor.register("identity_biased")
class IdentityBiasedCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def __init__(self, dim=None):
        self._dim = dim
    
    def compress(self, vector):
        dim = vector.shape[0]
        compressed_vector = CompressedVector(np.arange(dim), np.copy(vector), dim)
        return compressed_vector
    
    def alpha(self):
        return 1
    
    def num_nonzero_components(self):
        return self._dim


@FactoryCompressor.register("coordinate_sampling")
class CoordinateSamplingCompressor(UnbiasedBaseCompressor, SameSeedCompressor):
    INDEPENDENT = False
    RANDOM = True
    
    def __init__(self, node_number, total_number_of_nodes, seed, dim=None):
        self._generator = np.random.default_rng(seed=seed)
        self._dim = dim
        self._node_number = node_number
        self._total_number_of_nodes = total_number_of_nodes
    
    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        nodes_assignment = self._generator.integers(
            low=0, high=self._total_number_of_nodes, size=(self._dim,))
        mask = nodes_assignment == self._node_number
        sequence = np.arange(self._dim)
        indices = sequence[mask]
        values = vector[mask] * self._total_number_of_nodes
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector

    def num_nonzero_components(self):
        return float(self._dim) / self._total_number_of_nodes

    def omega(self):
        return self._total_number_of_nodes - 1


def _torch_generator(seed, is_cuda):
    device = 'cpu' if not is_cuda else 'cuda'
    generator_numpy = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(
        int(_generate_seed(generator_numpy)))
    return generator


class BaseRandKCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    def __init__(self, number_of_coordinates, dim=None):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim

    def compress(self, vector):
        raise NotImplementedError()
    
    def num_nonzero_components(self):
        return self._number_of_coordinates

    def omega(self):
        assert self._dim is not None
        return float(self._dim) / self._number_of_coordinates - 1


@FactoryCompressor.register("rand_k")
class RandKCompressor(BaseRandKCompressor):
    def __init__(self, number_of_coordinates, seed, dim=None):
        super(RandKCompressor, self).__init__(number_of_coordinates, dim)
        self._generator = np.random.default_rng(seed)

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates >= 0
        indices = self._generator.choice(dim, self._number_of_coordinates, replace = False)
        values = vector[indices] * float(dim / self._number_of_coordinates)
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector
    
    def copy(self):
        seed = np.random.default_rng()
        seed.bit_generator.state = self._generator.bit_generator.state
        return RandKCompressor(
            number_of_coordinates=self._number_of_coordinates, 
            seed=seed,
            dim=self._dim)


# TODO: omega slightly different
@FactoryCompressor.register("_rand_k_with_nat")
class RandKWithNatCompressor(RandKCompressor):
    def compress(self, vector):
        compressed_vector = super().compress(vector)
        indices, values, dim = compressed_vector.get_raw_info()
        out = _natural_compressor(values)
        return CompressedNatVector(indices, out, dim)


class CoreCompressedValues(object):
    def __init__(self, values, state, dim, number_of_coordinates):
        assert isinstance(values, np.ndarray) and values.ndim == 1
        self._state = state
        self._values = values
        self._dim = dim
        self._number_of_coordinates = number_of_coordinates
        
    def decompress(self):
        current_generator = np.random.default_rng()
        current_generator.bit_generator.state = self._state
        size = (self._number_of_coordinates, self._dim)
        gauss_vectors = current_generator.normal(size=size).astype(np.float32)
        decompressed_array = (gauss_vectors.T @ self._values) / float(self._number_of_coordinates)
        return decompressed_array
        
    def size_in_memory(self):
        return len(self._values) * self._values.itemsize * NUMBER_OF_BITES_IN_BYTE


@FactoryCompressor.register("core")
class CoreCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    
    def __init__(self, number_of_coordinates, seed, dim=None):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim
        self._generator = np.random.default_rng(seed)

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates >= 0
        state = copy.deepcopy(self._generator.bit_generator.state)
        gauss_vectors = self._generator.normal(size=(self._number_of_coordinates, dim)).astype(np.float32)
        compressed_vector = gauss_vectors @ vector
        compressed_vector = CoreCompressedValues(compressed_vector, state, dim, self._number_of_coordinates)
        return compressed_vector
    
    def num_nonzero_components(self):
        return self._number_of_coordinates

    def omega(self):
        assert self._dim is not None
        return float(self._dim + 1) / self._number_of_coordinates


@FactoryCompressor.register("biased_rand_k")
class BiasedRandKCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    def __init__(self, *args, **kwargs):
        self._compressor = RandKCompressor(*args, **kwargs)

    def compress(self, vector):
        unbiassed_compressed = self._compressor.compress(vector)
        omega = self._compressor.omega()
        unbiassed_compressed._values /= (omega + 1)
        return unbiassed_compressed

    def num_nonzero_components(self):
        return self._compressor.num_nonzero_components()
    
    def alpha(self):
        return 1 / (self._compressor.omega() + 1)


# TODO: alpha slightly different
@FactoryCompressor.register("_biased_rand_k_with_nat")
class BiasedRandKWithNatCompressor(BiasedRandKCompressor):
    def compress(self, vector):
        compressed_vector = super().compress(vector)
        indices, values, dim = compressed_vector.get_raw_info()
        values = (self._omega() + 1) * values
        out = _natural_compressor(values)
        total_omega = self._total_omega()
        out = out / (1 + total_omega)
        return CompressedNatVector(indices, out, dim)
    
    def _omega(self):
        omega = 1 / self.alpha() - 1
        return omega
    
    def _total_omega(self):
        omega = self._omega()
        omega_nat = 1 / 8. 
        total_omega = omega_nat * omega + omega_nat + omega
        return total_omega


@FactoryCompressor.register("rand_k_torch")
class RandKTorchCompressor(BaseRandKCompressor):
    def __init__(self, number_of_coordinates, seed, dim=None, is_cuda=False):
        super(RandKTorchCompressor, self).__init__(number_of_coordinates, dim)
        self._seed = seed
        self._is_cuda = is_cuda
        self._generator = _torch_generator(seed, is_cuda)
        self._compress_called = False

    def compress(self, vector):
        self._compress_called = True
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates >= 0
        indices = torch.randperm(
            dim, generator=self._generator, device=vector.device)[:self._number_of_coordinates]
        values = vector[indices] * float(dim / self._number_of_coordinates)
        compressed_vector = CompressedTorchVector(indices, values, dim)
        return compressed_vector
    
    def copy(self):
        assert not self._compress_called
        return RandKTorchCompressor(
            number_of_coordinates=self._number_of_coordinates, 
            seed=self._seed,
            dim=self._dim,
            is_cuda=self._is_cuda)


@FactoryCompressor.register("unbiased_top_k")
class UnbiasedTopKCompressor(UnbiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    ERROR = 1e-6
    def __init__(self, seed, dim=None):
        super(UnbiasedTopKCompressor, self).__init__()
        self._generator = np.random.default_rng(seed)
        self._dim = dim

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        vector_abs = np.abs(vector)
        l1_norm = np.sum(vector_abs)
        if l1_norm < self.ERROR:
            probs = None
        else:
            probs = vector_abs / l1_norm
        indices = self._generator.choice(dim, p=probs, size=1)
        values = l1_norm * (np.sign(vector[indices]))
        compressed_vector = CompressedVector(indices, values, dim)
        return compressed_vector
    
    def num_nonzero_components(self):
        return 1

    def omega(self):
        return float(self._dim) - 1


@FactoryCompressor.register("top_k")
class TopKCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def __init__(self, number_of_coordinates, dim=None):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates <= self._dim
        abs_vector = np.abs(vector)
        indices = abs_vector.argsort()[dim - self._number_of_coordinates:]
        values = vector[indices]
        return CompressedVector(indices, values, dim)

    def num_nonzero_components(self):
        return self._number_of_coordinates
    
    def alpha(self):
        return float(self._number_of_coordinates) / self._dim


@FactoryCompressor.register("top_k_torch")
class TopKTorchCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = False
    def __init__(self, number_of_coordinates, dim=None, is_cuda=False):
        self._number_of_coordinates = number_of_coordinates
        self._dim = dim

    def compress(self, vector):
        dim = vector.shape[0]
        assert self._dim is None or self._dim == dim
        self._dim = dim
        assert self._number_of_coordinates <= self._dim
        abs_vector = torch.abs(vector)
        indices = torch.argsort(abs_vector)[dim - self._number_of_coordinates:]
        values = vector[indices]
        return CompressedTorchVector(indices, values, dim)

    def num_nonzero_components(self):
        return self._number_of_coordinates
    
    def alpha(self):
        return float(self._number_of_coordinates) / self._dim


class BaseRoundsCompressor(BiasedBaseCompressor):
    INDEPENDENT = True
    RANDOM = True
    def __init__(self, compressor_name, compressor_params, dim=None, seed=None):
        assert 'seed' not in compressor_params
        if dim is not None:
            assert ('dim' not in compressor_params 
                    or compressor_params['dim'] is None
                    or compressor_params['dim'] == dim)
            compressor_params['dim'] = dim
        compressor_cls = FactoryCompressor.get(compressor_name)
        if compressor_cls.random():
            self._compressor = compressor_cls(seed=seed, **compressor_params)
        else:
            self._compressor = compressor_cls(**compressor_params)

    def _compress(self, vector, number_of_rounds):
        state = 0
        compressed_vectors = []
        for _ in range(number_of_rounds):
            compressed_vector = self._compressor.compress(vector - state)
            compressed_vectors.append(compressed_vector)
            decompressed_vector = compressed_vector.decompress()
            state = state + decompressed_vector
        return SumCompressedVectors(compressed_vectors)
    
    
@FactoryCompressor.register("rounds_compressor")
class RoundsCompressor(BaseRoundsCompressor):
    def __init__(self, number_of_rounds, *args, **kwargs):
        super(RoundsCompressor, self).__init__(*args, **kwargs)
        self._number_of_rounds = number_of_rounds

    def compress(self, vector):
        return self._compress(vector, self._number_of_rounds)


@FactoryCompressor.register("neolithic_compressor")
class NeolithicCompressor(RoundsCompressor):
    def __init__(self, *args, **kwargs):
        super(RoundsCompressor, self).__init__(*args, **kwargs)
        self._number_of_rounds = int(np.ceil(1 / self._compressor.alpha()))
    
    def compress(self, vector):
        return self._compress(vector, self._number_of_rounds)


def get_compressor_signatures(compressor_name, params, total_number_of_nodes, seed):
    return FactoryCompressor.get(compressor_name).get_compressor_signatures(
        params, total_number_of_nodes, seed)


def get_compressors(*args, **kwargs):
    signatures = get_compressor_signatures(*args, **kwargs)
    return [signature.create_instance() for signature in signatures]
