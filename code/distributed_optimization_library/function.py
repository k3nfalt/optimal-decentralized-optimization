import os
from enum import Enum
import itertools

from matplotlib.pyplot import axis
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.sparse
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.autograd import backward, grad
import torch.backends.cudnn as cudnn
import torchvision.models as models

from distributed_optimization_library.factory import Factory
from distributed_optimization_library.models.resnet import ResNet18, ResNetSmall

EPS = 1e-6


class FunctionType(Enum):
    NUMPY = 1
    TORCH_CPU = 2
    TORCH_CUDA = 3


class SamplingType(str, Enum):
    UNIFORM_WITH_REPLACEMENT = 'uniform_with_replacement'
    NICE = 'nice'
    IMPORTANCE = 'importance'


class BaseFunction(object):
    def __init__(self):
        self._statistics = {'gradient': 0}
    
    def value(self, point):
        raise NotImplementedError()
    
    def gradient(self, point):
        raise NotImplementedError()
    
    def dim(self):
        raise NotImplementedError()
    
    def liptschitz_gradient_constant(self):
        raise RuntimeError("liptschitz_gradient_constant is not defined.")
    
    @staticmethod
    def value_functions(functions, point):
        values = [function.value(point) for function in functions]
        return np.mean(values)
    
    def deterministic(self):
        return False
    
    def type(self):
        return FunctionType.NUMPY
    
    def statistics(self):
        return self._statistics
    
    def gradient_statistics(self, point):
        self._statistics['gradient'] += 1
        return self.gradient(point)


class BaseSeedFunction(BaseFunction):
    def __init__(self, seed):
        super(BaseSeedFunction, self).__init__()
        self._generator = np.random.default_rng(seed)


class BaseStochasticFunction(BaseSeedFunction):
    def __init__(self, seed):
        super(BaseStochasticFunction, self).__init__(seed)
    
    def stochastic_gradient(self, point):
        return self.stochastic_gradient_at_points((point,))[0]
    
    def stochastic_gradient_at_points(self, points):
        raise NotImplementedError()


# todo(tyurina): Inherit it from BaseStochasticFunction
class BaseMeanFunction(BaseSeedFunction):
    def __init__(self, seed, number_of_functions):
        super(BaseMeanFunction, self).__init__(seed)
        self._number_of_functions = number_of_functions
        self._statistics['batch_gradient_at_points'] = 0
    
    def number_of_functions(self):
        return self._number_of_functions
    
    def batch_gradient(self, point, batch_size):
        return self.batch_gradient_at_points((point,), batch_size=batch_size)[0]
    
    def batch_gradient_at_points(self, points, batch_size):
        raise NotImplementedError()
    
    def batch_gradient_at_points_with_indices(self, points, indices):
        raise NotImplementedError()
    
    def liptschitz_max_gradient_constant(self):
        raise RuntimeError("liptschitz_max_gradient_constant is not defined.")
    
    def batch_gradient_at_points_statistics(self, points, batch_size):
        self._statistics['batch_gradient_at_points'] += len(points) * batch_size
        return self.batch_gradient_at_points(points, batch_size)


class OptimizationProblemMeta(object):    
    def is_convex(self):
        raise RuntimeError("is_convex is not defined.")
    
    def strongly_convex_constant(self):
        assert self.is_convex()
        return 0.0
    
    def bregman_smoothness_variance(self):
        return None
    
    def smoothness_variance(self):
        return None


def generate_random_vector(dim, seed):
    generator = np.random.default_rng(seed)
    return generator.random((dim,), dtype=np.float32)


def generate_random_nonegative_symmetric_matrix(dim, seed, reg=None):
    generator = np.random.default_rng(seed)
    A = 2 * generator.random((dim, dim), dtype=np.float32) - 1
    B = np.dot(A, A.transpose())
    if reg is not None:
        B = B + reg * np.eye(dim, dtype=np.float32)
    return B


def create_worst_case(dim, liptschitz_gradient_constant, noise_lambda=0, seed=None,
                      strategy='mul'):
    scale = liptschitz_gradient_constant / 4.
    main_diag = 2 * np.ones((dim,), dtype=np.float32)
    side_diag = -1 * np.ones((dim - 1,), dtype=np.float32)
    b = np.zeros((dim,), dtype=np.float32)
    b[0] = -1
    if noise_lambda > 0:
        generator = np.random.default_rng(seed)
        if strategy == 'add':
            noise = noise_lambda * generator.standard_exponential(size=(1,), dtype=np.float32)
            main_diag += noise
            side_diag += noise
        if strategy == 'mul':
            noise_scale = 1 + noise_lambda * generator.standard_normal(size=(1,), dtype=np.float32)
            noise_bias = noise_lambda * generator.standard_normal(size=(1,), dtype=np.float32)
            b[0] += noise_bias
            b[0] *= noise_scale
            main_diag *= noise_scale
            side_diag *= noise_scale
    b[0] *= scale
    return scale * main_diag, scale * side_diag, b


class QuadraticFunction(BaseFunction):
    '''
    Function f(x) = 1/2 x^T A x - b^T x
    '''
    def __init__(self, A, b, check=False):
        super(QuadraticFunction, self).__init__()
        if check:
            self._check(A)
        self._A = A
        self._b = b
    
    def value(self, point):
        return (1/2.) * np.dot(point, np.dot(self._A, point)) - np.dot(self._b, point)
        
    def gradient(self, point):
        return np.dot(self._A, point) - self._b
    
    def dim(self):
        return self._A.shape[1]
    
    def liptschitz_gradient_constant(self):
        svdvals = scipy.linalg.svdvals(self._A)
        return np.max(svdvals)
    
    def _check(self, A):
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError as err:
            raise RuntimeError("Something wrong with matrix")
    
    @staticmethod
    def create_random(dim, seed=None, reg=None):
        generator = np.random.default_rng(seed=seed)
        return QuadraticFunction(generate_random_nonegative_symmetric_matrix(dim, generator, reg),
                                 generate_random_vector(dim, generator))
        
    @staticmethod
    def create_norm_square(dim):
        return QuadraticFunction(np.eye(dim, dtype=np.float32), np.zeros(dim, dtype=np.float32))

    @staticmethod
    def smoothness_variance_bound_functions(functions):
        for function in functions:
            assert isinstance(function, QuadraticFunction)
        matrices = [function._A for function in functions]
        
        lambda_square_matrix = lambda A: np.linalg.matrix_power(A, 2)
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        
        mean_matrices = lambda_mean_matrices(matrices)
        square_mean_matrices = lambda_square_matrix(mean_matrices)
        square_matrices = list(map(lambda_square_matrix, matrices))
        mean_square_matrices = lambda_mean_matrices(square_matrices)
        svdvals = scipy.linalg.svdvals(mean_square_matrices - square_mean_matrices)
        op_norm = np.max(svdvals)
        svb = np.sqrt(op_norm)
        return svb

    @staticmethod
    def liptschitz_gradient_constant_functions(functions):
        for function in functions:
            assert isinstance(function, QuadraticFunction)
        matrices = [function._A for function in functions]
        
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        
        mean_matrices = lambda_mean_matrices(matrices)
        svdvals = scipy.linalg.svdvals(mean_matrices)
        op_norm = np.max(svdvals)
        return op_norm

    @staticmethod
    def eigenvalues_functions(functions):
        for function in functions:
            assert isinstance(function, QuadraticFunction)
        matrices = [function._A for function in functions]
        
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        
        mean_matrices = lambda_mean_matrices(matrices)
        eigvals = scipy.linalg.eigvals(mean_matrices)
        return eigvals

    @classmethod
    def min_eigenvalue_functions(cls, functions):
        eigvals = cls.eigenvalues_functions(functions)
        return np.min(eigvals)
    
    @staticmethod
    def analytical_solution_functions(functions):
        for function in functions:
            assert isinstance(function, QuadraticFunction)
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        solution = np.linalg.solve(lambda_mean_matrices([function._A for function in functions]),
                                   lambda_mean_matrices([function._b for function in functions]))
        return solution


class StochasticQuadraticFunction(BaseStochasticFunction):
    def __init__(self, A, b, seed, noise=1.0):
        super(StochasticQuadraticFunction, self).__init__(seed)
        self._quadratic_function = QuadraticFunction(A, b)
        self._noise = noise
        
    def dim(self):
        return self._quadratic_function.dim()
    
    def stochastic_gradient_at_points(self, points):
        noise = self._noise * self._generator.normal()
        stochastic_gradients = []
        for point in points:
            stochastic_gradient = self._quadratic_function.gradient(point)
            stochastic_gradient = stochastic_gradient + noise
            stochastic_gradients.append(stochastic_gradient)
        return stochastic_gradients
    
    def gradient(self, point):
        return self._quadratic_function.gradient(point)

    @staticmethod
    def create_random(dim, seed=None, reg=None, noise=1.0):
        generator = np.random.default_rng(seed=seed)
        A = generate_random_nonegative_symmetric_matrix(dim, generator, reg)
        b = generate_random_vector(dim, generator)
        return StochasticQuadraticFunction(A, b, seed=generator, noise=noise)


class BaseMeanQuadraticFunction(BaseMeanFunction, BaseStochasticFunction):
    def __init__(self, quadratic_functions, seed, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT, batch_size=1):
        super(BaseMeanQuadraticFunction, self).__init__(seed, len(quadratic_functions))
        self._quadratic_functions = quadratic_functions
        self._sampling = sampling
        self._batch_size = batch_size
        if self._sampling == SamplingType.IMPORTANCE:
            self._local_liptschitz = [qf.liptschitz_gradient_constant() for qf in self._quadratic_functions]
            self._importance_probs = self._local_liptschitz / np.sum(self._local_liptschitz)
    
    def dim(self):
        return self._quadratic_functions[0].dim()
    
    def get_quadratic_functions(self):
        return self._quadratic_functions
    
    def value(self, point):
        sum_value = 0
        for quadratic_function in self._quadratic_functions:
            sum_value += quadratic_function.value(point)
        return sum_value / float(self._number_of_functions)
    
    def gradient(self, point):
        sum_gradients = 0
        for quadratic_function in self._quadratic_functions:
            sum_gradients += quadratic_function.gradient(point)
        return sum_gradients / float(self._number_of_functions)
    
    def batch_gradient_at_points(self, points, batch_size):
        if self._sampling == SamplingType.UNIFORM_WITH_REPLACEMENT:
            indices = self._generator.integers(self._number_of_functions, size=batch_size)
        elif self._sampling == SamplingType.NICE:
            indices = self._generator.permutation(self._number_of_functions)[:batch_size]
        elif self._sampling == SamplingType.IMPORTANCE:
            indices = self._generator.choice(self._number_of_functions, size=batch_size,
                                             p=self._importance_probs)
        else:
            raise NotImplementedError()
        return self.batch_gradient_at_points_with_indices(points, indices)

    def batch_gradient_at_points_with_indices(self, points, indices):
        output_gradients = []
        for point in points:
            output_gradient = 0
            for index in indices:
                if self._sampling == SamplingType.NICE:
                    unbiased_sample = self._quadratic_functions[index].gradient(point)
                    output_gradient = output_gradient + unbiased_sample
                if self._sampling == SamplingType.UNIFORM_WITH_REPLACEMENT:
                    unbiased_sample = self._quadratic_functions[index].gradient(point)
                    output_gradient = output_gradient + unbiased_sample
                if self._sampling == SamplingType.IMPORTANCE:
                    unbiased_sample = self._quadratic_functions[index].gradient(point) / \
                        (self._number_of_functions * self._importance_probs[index])
                    output_gradient = output_gradient + unbiased_sample
            assert output_gradient.size == point.size
            output_gradients.append(output_gradient / len(indices))
        return output_gradients

    def stochastic_gradient_at_points(self, points):
        return self.batch_gradient_at_points(points, self._batch_size)

    def liptschitz_max_gradient_constant(self):
        return np.max([quadratic_function.liptschitz_gradient_constant() 
                       for quadratic_function in self._quadratic_functions])
    
    def liptschitz_gradient_constant(self):
        return self.liptschitz_max_gradient_constant()  # pessimistic 


class MeanQuadraticFunction(BaseMeanQuadraticFunction):
    def __init__(self, As, bs, seed, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT):
        super(MeanQuadraticFunction, self).__init__(
            [QuadraticFunction(A, b) for A, b in zip(As, bs)], seed, sampling)
        
    @staticmethod
    def create_random(dim, number_of_functions, seed=None, reg=None):
        generator = np.random.default_rng(seed=seed)
        As = [generate_random_nonegative_symmetric_matrix(dim, generator, reg) for _ in range(number_of_functions)]
        bs = [generate_random_vector(dim, generator) for _ in range(number_of_functions)]
        return MeanQuadraticFunction(As, bs, generator)


class TridiagonalQuadraticFunction(BaseFunction):
    def __init__(self, main_diag, side_diag, b):
        super(TridiagonalQuadraticFunction, self).__init__()
        self._main_diag = main_diag
        self._side_diag = side_diag
        self._A = scipy.sparse.diags([side_diag, main_diag, side_diag], [-1, 0, 1])
        self._b = b
        
    def get_sparse_data(self):
        return self._main_diag, self._side_diag, self._b
    
    def value(self, point):
        return (1/2.) * np.dot(point, self._A.dot(point)) - np.dot(self._b, point)
        
    def gradient(self, point):
        return self._A.dot(point) - self._b
    
    def dim(self):
        return self._A.shape[1]
    
    def liptschitz_gradient_constant(self):
        # if (np.isclose(np.max(self._main_diag), np.min(self._main_diag)) and
        #     np.isclose(np.max(self._side_diag), np.min(self._side_diag))):
        #     return self._main_diag[0] + 2 * np.abs(self._side_diag[0]) * np.cos(np.pi / (len(self._main_diag) + 1))
        # The above formula should be fixed
        eigvals = scipy.linalg.eigh_tridiagonal(
            self._main_diag, self._side_diag, eigvals_only=True)
        return np.max(np.abs(eigvals))
    
    def to_quadratic_function(self):
        return QuadraticFunction(self._A.todense(), self._b)

    @staticmethod
    def create_worst_case_functions(num_nodes, dim, liptschitz_gradient_constant=1, seed=None, noise_lambda=0,
                                    strongly_convex_constant=1e-5):
        main_diags = []
        side_diags = []
        bs = []
        for _ in range(num_nodes):
            main_diag, side_diag, b = create_worst_case(dim, liptschitz_gradient_constant, noise_lambda, seed)
            main_diags.append(main_diag)
            side_diags.append(side_diag)
            bs.append(b)
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        mean_main_diag = lambda_mean_matrices(main_diags)
        mean_side_diag = lambda_mean_matrices(side_diags)
        eigs = scipy.linalg.eigh_tridiagonal(mean_main_diag, mean_side_diag, eigvals_only=True)
        min_eig = np.min(eigs)
        funcs = []
        for main_diag, side_diag, b in zip(main_diags, side_diags, bs):
            main_diag = main_diag - min_eig
            main_diag = main_diag + strongly_convex_constant
            funcs.append(TridiagonalQuadraticFunction(main_diag, side_diag, b))
        return funcs
    
    @staticmethod
    def create_convex_different_liptschitz(num_nodes, dim, liptschitz_gradient_constant=1, seed=None, noise_lambda=0):
        generator = np.random.default_rng(seed)
        funcs = []
        for _ in range(num_nodes):
            main_diag, side_diag, b = create_worst_case(dim, liptschitz_gradient_constant, noise_lambda=0.0, seed=seed)
            scale = 1 + noise_lambda * generator.standard_exponential(size=(1,), dtype=np.float32)
            main_diag *= scale
            side_diag *= scale
            b[0] += noise_lambda * generator.standard_normal(size=(1,), dtype=np.float32)
            funcs.append(TridiagonalQuadraticFunction(main_diag, side_diag, b))
        return funcs
    
    @staticmethod
    def eigenvalues_functions(functions):
        for function in functions:
            assert isinstance(function, TridiagonalQuadraticFunction)
        main_diags = [function._main_diag for function in functions]
        side_diags = [function._side_diag for function in functions]
        
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        
        mean_main_diag = lambda_mean_matrices(main_diags)
        mean_side_diag = lambda_mean_matrices(side_diags)
        eigvals = scipy.linalg.eigh_tridiagonal(mean_main_diag, mean_side_diag, eigvals_only=True)
        return eigvals
    
    @classmethod
    def min_eigenvalue_functions(cls, functions):
        eigvals = cls.eigenvalues_functions(functions)
        return np.min(eigvals)
    
    @classmethod
    def liptschitz_gradient_constant_functions(cls, functions):
        eigvals = cls.eigenvalues_functions(functions)
        return np.max(np.abs(eigvals))
    
    @staticmethod
    def _smoothness_variance_bound_functions(functions, weights=None, subtract_mean=True):
        if weights is None:
            weights = np.ones(len(functions), dtype=np.float32) / len(functions)
        for function in functions:
            assert isinstance(function, TridiagonalQuadraticFunction)
        matrices = [function._A for function in functions]
        
        lambda_square_matrix = lambda A: A.dot(A)
        def lambda_mean_matrices(A_list, weights=None):
            if weights is not None:
                A_list_weights = [A / (len(A_list) * weight) for A, weight in zip(A_list, weights)]
            else:
                A_list_weights = A_list
            return sum(A_list_weights) / len(A_list)
        
        mean_matrices = lambda_mean_matrices(matrices)
        square_mean_matrices = lambda_square_matrix(mean_matrices)
        square_matrices = list(map(lambda_square_matrix, matrices))
        mean_square_matrices = lambda_mean_matrices(square_matrices, weights)
        if subtract_mean:
            result_matrix = mean_square_matrices - square_mean_matrices
        else:
            result_matrix = mean_square_matrices
        svdvals = scipy.sparse.linalg.svds(result_matrix, k=1, return_singular_vectors=False)
        op_norm = np.max(svdvals)
        svb = np.sqrt(op_norm)
        return svb
    
    @classmethod
    def smoothness_variance_bound_functions(cls, functions, weights=None):
        return cls._smoothness_variance_bound_functions(functions, weights)
    
    @classmethod
    def liptschitz_gradient_constant_plus_functions(cls, functions, weights=None):
        return cls._smoothness_variance_bound_functions(functions, weights, subtract_mean=False)
    
    @staticmethod
    def analytical_solution_functions(functions):
        for function in functions:
            assert isinstance(function, TridiagonalQuadraticFunction)
        lambda_mean_matrices = lambda A_list: sum(A_list) / len(A_list)
        solution = np.linalg.solve(lambda_mean_matrices([function._A for function in functions]).todense(),
                                   lambda_mean_matrices([function._b for function in functions]))
        return solution
    
    @staticmethod
    def dump_functions(functions, folder):
        assert not os.path.exists(folder)
        os.mkdir(folder)
        for index, function in enumerate(functions):
            function_path = os.path.join(folder, "function_{}".format(index))
            os.mkdir(function_path)
            np.save(os.path.join(function_path, 'main_diag'), function._main_diag)
            np.save(os.path.join(function_path, 'side_diag'), function._side_diag)
            np.save(os.path.join(function_path, 'b'), function._b)
    
    @staticmethod
    def load_functions(folder):
        assert os.path.exists(folder)
        num_functions = len(os.listdir(folder))
        functions = []
        for index in range(num_functions):
            function_path = os.path.join(folder, "function_{}".format(index))
            main_diag = np.load(os.path.join(function_path, 'main_diag.npy'))
            side_diag = np.load(os.path.join(function_path, 'side_diag.npy'))
            b = np.load(os.path.join(function_path, 'b.npy'))
            function = TridiagonalQuadraticFunction(main_diag, side_diag, b)
            functions.append(function)
        return functions


class MeanTridiagonalQuadraticFunction(BaseMeanQuadraticFunction):
    def __init__(self, main_diags, side_diags, bs,
                 seed=None, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT):
        quadratic_functions = [TridiagonalQuadraticFunction(main_diag, side_diag, b) 
                               for main_diag, side_diag, b in zip(main_diags, side_diags, bs)]
        super(MeanTridiagonalQuadraticFunction, self).__init__(quadratic_functions, seed, sampling)
    
    @classmethod
    def from_tridiagonal_quadratic_functions(cls, quadratic_functions,
                                             seed=None, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT):
        params = [qf.get_sparse_data() for qf in quadratic_functions]
        main_diags, side_diags, bs = list(zip(*params))
        return cls(main_diags, side_diags, bs, seed=seed, sampling=sampling)
        
    @staticmethod
    def create_worst_case_functions(num_of_functions, dim, liptschitz_gradient_constant=1, 
                                    seed=None, noise_lambda=0, strongly_convex_constant=1e-5,
                                    sampling=SamplingType.UNIFORM_WITH_REPLACEMENT):
        quadratic_functions = TridiagonalQuadraticFunction.create_worst_case_functions(
            num_of_functions, dim, liptschitz_gradient_constant,
            seed, noise_lambda, strongly_convex_constant)
        return MeanTridiagonalQuadraticFunction.from_tridiagonal_quadratic_functions(
            quadratic_functions, seed=seed, sampling=sampling)
    
    @staticmethod
    def create_convex_different_liptschitz(num_of_functions, dim, liptschitz_gradient_constant=1, 
                                           seed=None, noise_lambda=0, strongly_convex_constant=1e-5,
                                           sampling=SamplingType.UNIFORM_WITH_REPLACEMENT):
        quadratic_functions = TridiagonalQuadraticFunction.create_convex_different_liptschitz(
            num_of_functions, dim, liptschitz_gradient_constant,
            seed, noise_lambda, strongly_convex_constant)
        return MeanTridiagonalQuadraticFunction.from_tridiagonal_quadratic_functions(
            quadratic_functions, seed=seed, sampling=sampling)

    def dump(self, folder):
        TridiagonalQuadraticFunction.dump_functions(self._quadratic_functions, folder)
    
    @classmethod
    def load(cls, folder, seed=None, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT):
        quadratic_functions = TridiagonalQuadraticFunction.load_functions(folder)
        return cls.from_tridiagonal_quadratic_functions(
            quadratic_functions, seed=seed, sampling=sampling)


class StochasticTridiagonalQuadraticFunction(BaseStochasticFunction):
    def __init__(self, main_diag, side_diag, b, seed, noise, type_noise='add'):
        super(StochasticTridiagonalQuadraticFunction, self).__init__(seed)
        self._tridiagonal_quadratic = TridiagonalQuadraticFunction(main_diag, side_diag, b)
        self._noise = noise
        self._type_noise = type_noise
    
    @staticmethod
    def from_tridiagonal_quadratic(tridiagonal_quadratic, seed, noise, type_noise='add'):
        return StochasticTridiagonalQuadraticFunction(
            tridiagonal_quadratic._main_diag,
            tridiagonal_quadratic._side_diag,
            tridiagonal_quadratic._b, seed, noise, type_noise)
        
    def dim(self):
        return self._tridiagonal_quadratic.dim()
    
    def stochastic_gradient_at_points(self, points):
        if self._type_noise == 'add':
            noise = self._noise * self._generator.normal()
            stochastic_gradients = []
            for point in points:
                stochastic_gradient = self._tridiagonal_quadratic.gradient(point)
                stochastic_gradient = stochastic_gradient + noise
                stochastic_gradients.append(stochastic_gradient)
            return stochastic_gradients
        elif self._type_noise == 'zero-last':
            assert len(points) == 1
            point = points[0]
            def bernoulli_sample(random_generator, prob):
                assert prob >= 0 and prob <= 1.0
                if prob == 0.0:
                    return False
                return random_generator.random() < prob
            coin = bernoulli_sample(self._generator, self._noise)
            scale = coin / self._noise
            def prog(x):
                return int(np.sort(np.nonzero(x)[0])[-1])
            prog_point = prog(point)
            scale_from = prog_point + 1
            stochastic_gradient = self._tridiagonal_quadratic.gradient(point)
            # print("Var: ", np.abs(stochastic_gradient[scale_from]), "Prog: ", prog_point, "Len: ", len(stochastic_gradient))
            if scale_from < len(stochastic_gradient):
                stochastic_gradient[scale_from:] = stochastic_gradient[scale_from:] * scale
            return [stochastic_gradient]
        else:
            raise RuntimeError()
    
    def value(self, point):
        return self._tridiagonal_quadratic.value(point)
    
    def gradient(self, point):
        return self._tridiagonal_quadratic.gradient(point)


class StochasticMatrixTridiagonalQuadraticFunction(BaseStochasticFunction):
    def __init__(self, main_diag, side_diag, b, seed, noise):
        super(StochasticMatrixTridiagonalQuadraticFunction, self).__init__(seed)
        self._main_diag = main_diag
        self._side_diag = side_diag
        self._b = b
        self._tridiagonal_quadratic = TridiagonalQuadraticFunction(main_diag, side_diag, b)
        self._noise = noise
    
    @staticmethod
    def from_tridiagonal_quadratic(tridiagonal_quadratic, seed, noise):
        return StochasticMatrixTridiagonalQuadraticFunction(
            tridiagonal_quadratic._main_diag,
            tridiagonal_quadratic._side_diag,
            tridiagonal_quadratic._b, seed, noise)
        
    def dim(self):
        return len(self._main_diag)
    
    def stochastic_gradient_at_points(self, points):
        noise = self._noise * self._generator.normal()
        main_diag = self._main_diag + noise
        stochastic_tridiagonal_quadratic = \
            TridiagonalQuadraticFunction(main_diag, self._side_diag, self._b)
        stochastic_gradients = []
        for point in points:
            stochastic_gradient = stochastic_tridiagonal_quadratic.gradient(point)
            stochastic_gradients.append(stochastic_gradient)
        return stochastic_gradients
    
    def value(self, point):
        return self._tridiagonal_quadratic.value(point)
    
    def gradient(self, point):
        return self._tridiagonal_quadratic.gradient(point)


class QuadraticOptimizationProblemMeta(OptimizationProblemMeta):
    def __init__(self, functions):
        self._functions = functions
        self._min_eigenvalue = None
    
    def is_convex(self):
        self._min_eigenvalue = self._min_eigenvalue or \
            QuadraticFunction.min_eigenvalue_functions(self._functions)
        assert self._min_eigenvalue.imag == 0
        self._min_eigenvalue = self._min_eigenvalue.real 
        return self._min_eigenvalue >= -EPS
    
    def strongly_convex_constant(self):
        assert self.is_convex()
        return self._min_eigenvalue


class TridiagonalQuadraticOptimizationProblemMeta(OptimizationProblemMeta):
    def __init__(self, functions, worst_smoothness_variance=True):
        self._functions = functions
        self._min_eigenvalue = None
        self._worst_smoothness_variance = worst_smoothness_variance
    
    def is_convex(self):
        self._min_eigenvalue = self._min_eigenvalue or \
            TridiagonalQuadraticFunction.min_eigenvalue_functions(self._functions)
        return self._min_eigenvalue >= -EPS
    
    def strongly_convex_constant(self):
        assert self.is_convex()
        return self._min_eigenvalue
    
    def smoothness_variance(self):
        if self._worst_smoothness_variance:
            return super(TridiagonalQuadraticOptimizationProblemMeta, self).smoothness_variance()
        else:
            return TridiagonalQuadraticFunction.smoothness_variance_bound_functions(self._functions)


class QuadraticTorchFunction(BaseFunction):
    def __init__(self, A, b):
        super(QuadraticTorchFunction, self).__init__()
        self._check(A)
        self._A = A
        self._b = b
    
    def value(self, point):
        return (1/2.) * torch.dot(point, torch.mv(self._A, point)) - torch.dot(self._b, point)
        
    def gradient(self, point):
        point = torch.tensor(point, requires_grad=True)
        value = self.value(point)
        value.backward()
        return point.grad
    
    def dim(self):
        return self._A.shape[1]
    
    def liptschitz_gradient_constant(self):
        return self._liptschitz_gradient_constant
    
    def cuda(self):
        A = self._A.cuda()
        b = self._b.cuda()
        return QuadraticTorchFunction(A, b)
    
    def _check(self, A):
        A = A.cpu().numpy()
        try:
            np.linalg.cholesky(A)
        except np.linalg.LinAlgError as err:
            raise RuntimeError("Something wrong with matrix")
        self._eigvals = np.linalg.eigvals(A)
        self._liptschitz_gradient_constant = np.max(self._eigvals)
        
    @staticmethod
    def create_random(dim, seed=None):
        generator = np.random.default_rng(seed=seed)
        return QuadraticTorchFunction(torch.tensor(generate_random_nonegative_symmetric_matrix(dim, generator)),
                                      torch.tensor(generate_random_vector(dim, generator)))
    
    def type(self):
        if not self._A.is_cuda:
            return FunctionType.TORCH_CPU
        else:
            return FunctionType.TORCH_CUDA


def parameters_to_array(parameters, grad=False):
    flatten_parameters = []
    for parameter in parameters:
        if grad:
            parameter_np = parameter.grad.cpu().numpy().flatten()
        else:
            parameter_np = parameter.detach().cpu().numpy().flatten()
        flatten_parameters.append(parameter_np)
    return np.concatenate(flatten_parameters)


def array_to_parameters(parameters, array):
    shift = 0
    for parameter in parameters:
        num_elements = parameter.numel()
        parameter_torch = torch.from_numpy(array[shift:shift + num_elements])
        parameter_torch = parameter_torch.reshape(parameter.data.shape)
        if parameter.data.is_cuda:
            parameter.data.copy_(parameter_torch.cuda())
        else:
            parameter.data.copy_(parameter_torch)
        shift += num_elements


def flatten_parameters_to_parameters(parameters, flatten_parameters):
    for parameter, flatten_parameter in zip(parameters, flatten_parameters):
        parameter.data.flatten().copy_(flatten_parameter.data)


def parameters_to_flatten_parameters(parameters, grad=False, clone=True):
    flatten_parameters = []
    for parameter in parameters:
        if grad:
            parameter = parameter.grad.detach().flatten()
        else:
            parameter = parameter.detach().flatten()
        if clone:
            parameter = parameter.clone()
        flatten_parameters.append(parameter)
    return flatten_parameters


def tensor_to_parameters(parameters, tensor):
    shift = 0
    for parameter in parameters:
        num_elements = parameter.numel()
        parameter.data.flatten().copy_(tensor[shift:shift + num_elements])
        shift += num_elements


def parameters_to_tensor(parameters, grad=False):
    flatten_parameters = parameters_to_flatten_parameters(parameters, grad=grad, clone=False)
    return torch.cat(flatten_parameters)


class TwoLayerNeuralNet(nn.Module):
    def __init__(self, input_dim, number_of_classes, neural_network_name):
        super(TwoLayerNeuralNet, self).__init__()
        assert neural_network_name is not None
        self.neural_network_name = neural_network_name
        if neural_network_name != 'two_layer_neural_net_linear':
            if neural_network_name == 'two_layer_neural_net_worst_case_sigmoid':
                self.A = torch.nn.Parameter(torch.zeros(1, input_dim, 10) - 3)
                self.fc = nn.Linear(input_dim, number_of_classes)
            elif neural_network_name == 'two_layer_neural_net_worst_case':
                d = 8
                self.fc1 = nn.Linear(input_dim, d, bias=False)
                layers = [nn.Linear(d, d, bias=False) for _ in range(10)]
                layers.append(nn.Linear(d, number_of_classes))
                self.fc2 = nn.Sequential(*layers)
                self.activation = nn.Identity()
            else:
                self.fc1 = nn.Linear(input_dim, 32)
                if neural_network_name == 'two_layer_neural_net':
                    self.activation = nn.Sigmoid()
                elif neural_network_name == 'two_layer_neural_net_relu':
                    self.activation = nn.ReLU()
                elif neural_network_name == 'two_layer_neural_net_skip_connection':
                    self.activation = nn.Sigmoid()
                    self.fc3 = nn.Linear(32, 32)
                self.fc2 = nn.Linear(32, number_of_classes)
        else:
            self.fc1 = nn.Linear(input_dim, number_of_classes)
        
    def forward(self, x, test=False):
        if self.neural_network_name == 'two_layer_neural_net_worst_case_sigmoid':
            x = x.reshape(x.shape[0], x.shape[1], 1)
            x = x + self.A
            x = torch.sigmoid(x)
            x = torch.sum(x, axis=-1)
            x = self.fc(x)
            return x
        else:
            if self.neural_network_name != 'two_layer_neural_net_linear':
                x1 = self.fc1(x)
                x2 = self.activation(x1)
                if self.neural_network_name == 'two_layer_neural_net_skip_connection':
                    x2 = self.fc3(x2) + x1
                x3 = self.fc2(x2)
            else:
                x3 = self.fc1(x)
            return x3


class NeuralNetworkFunction(BaseFunction):
    def __init__(self, features, labels, number_of_classes=2, is_cuda=False, reg_paramterer=0.0,
                 neural_network_name=None):
        super(NeuralNetworkFunction, self).__init__()
        self._features = torch.tensor(features)
        self._labels_np = labels
        self._labels = torch.tensor(labels)
        self._input_dim = features.shape[1]
        self._number_of_classes = number_of_classes
        self._nn = TwoLayerNeuralNet(self._input_dim, self._number_of_classes, neural_network_name)
        self._is_cuda = is_cuda
        self._reg_paramterer = reg_paramterer
        if self._is_cuda:
            self._features = self._features.cuda()
            self._labels = self._labels.cuda()
            self._nn = self._nn.cuda()
        self._criterion = nn.CrossEntropyLoss()
    
    def value(self, point):
        return self._loss(point).cpu().detach().numpy()
    
    def gradient(self, point):
        self._nn.zero_grad()
        loss = self._loss(point)
        loss.backward()
        return parameters_to_array(self._nn.parameters(), grad=True)
    
    def dim(self):
        parameters_flatten = parameters_to_array(self._nn.parameters())
        return parameters_flatten.shape[0]
    
    def get_current_point(self):
        return parameters_to_array(self._nn.parameters())
        
    def _loss(self, point):
        logits = self._logits(point)
        loss = self._criterion(logits, self._labels)
        if self._reg_paramterer > 0.0:
            all_parameters = []
            for parameter in self._nn.parameters():
                all_parameters.append(parameter.flatten())
            all_parameters = torch.cat(all_parameters)
            reg = torch.std(all_parameters)
            loss = loss + self._reg_paramterer * reg
                
        return loss
    
    def _logits(self, point):
        array_to_parameters(self._nn.parameters(), point)
        logits = self._nn(self._features)
        return logits

    def _check_accuracy(self, point):
        logits = self._logits(point)
        prediction = np.argmax(logits.detach().numpy(), axis=1)
        acc = np.sum(prediction == self._labels_np) / float(len(self._labels_np))
        return acc


class AutoEncoderNeuralNet(nn.Module):
    def __init__(self, input_dim, encode_dim, point_initializer=None):
        super(AutoEncoderNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, encode_dim, bias=False)
        self.fc2 = nn.Linear(encode_dim, input_dim, bias=False)
        if point_initializer is not None:
            if point_initializer == 'xavier_normal':
                torch.nn.init.xavier_normal_(self.fc1.weight)
                torch.nn.init.xavier_normal_(self.fc2.weight)
            else:
                raise RuntimeError()
        
    def forward(self, x, test=False):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class AutoEncoderEqualNeuralNet(nn.Module):
    def __init__(self, input_dim, encode_dim):
        super(AutoEncoderEqualNeuralNet, self).__init__()
        self._A = torch.nn.Parameter(nn.Linear(input_dim, encode_dim, bias=False).weight)
        
    def forward(self, x, test=False):
        x = torch.mm(x, self._A.t())
        x = torch.mm(x, self._A)
        return x


class AutoEncoderNeuralNetworkFunction(BaseStochasticFunction):
    def __init__(self, features, labels, neural_network_name, reg_paramterer=0.0,
                 point_initializer=None, batch_size=None, seed=None, encode_dim=16):
        super(AutoEncoderNeuralNetworkFunction, self).__init__(seed=seed)
        self._features = torch.tensor(features)
        self._input_dim = features.shape[1]
        self._neural_network_name = neural_network_name
        if self._neural_network_name == 'auto_encoder':
            self._nn = AutoEncoderNeuralNet(self._input_dim, encode_dim, point_initializer)
        elif self._neural_network_name == 'auto_encoder_equal':
            self._nn = AutoEncoderEqualNeuralNet(self._input_dim, encode_dim)
        self._criterion = nn.MSELoss()
        self._reg_paramterer = reg_paramterer
        self._batch_size = batch_size
    
    def value(self, point):
        return self._loss(point, self._features).cpu().detach().numpy()
    
    def gradient(self, point):
        self._nn.zero_grad()
        loss = self._loss(point, self._features)
        loss.backward()
        return parameters_to_array(self._nn.parameters(), grad=True)
    
    def stochastic_gradient_at_points(self, points):
        assert len(points) == 1
        point = points[0]
        assert self._batch_size is not None
        assert len(self._features.shape) == 2
        indices = self._generator.integers(self._features.shape[0], size=self._batch_size)
        features = self._features[indices, :]
        self._nn.zero_grad()
        loss = self._loss(point, features)
        loss.backward()
        return [parameters_to_array(self._nn.parameters(), grad=True)]
    
    def dim(self):
        parameters_flatten = parameters_to_array(self._nn.parameters())
        return parameters_flatten.shape[0]
    
    def get_current_point(self):
        return parameters_to_array(self._nn.parameters())
        
    def _loss(self, point, features):
        outputs = self._outputs(point, features)
        loss = self._criterion(outputs, features)
        if self._reg_paramterer > 0.0:
            if self._neural_network_name == 'auto_encoder':
                a = torch.transpose(self._nn.fc1.weight, 0, 1)
                b = torch.transpose(self._nn.fc2.weight, 0, 1)
                n = a.shape[0]
                reg = torch.linalg.matrix_norm(torch.mm(a, b) - torch.eye(n)) ** 2
            elif self._neural_network_name == 'auto_encoder_equal':
                n = self._nn._A.shape[1]
                reg = torch.linalg.matrix_norm(torch.mm(self._nn._A.t(), self._nn._A) - torch.eye(n)) ** 2
            loss = loss + self._reg_paramterer * reg
        return loss
    
    def _outputs(self, point, features):
        array_to_parameters(self._nn.parameters(), point)
        outputs = self._nn(features)
        return outputs
    
    def _check_accuracy(self, point):
        outputs = self._outputs(point, self._features)
        loss = self._criterion(outputs, self._features)
        return float(loss.detach().numpy())


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Resnet18Function(BaseStochasticFunction, BaseMeanFunction):
    _LARGE_NUMBER = 10**12
    def __init__(self, dataset, batch_size, seed, num_workers=None, is_cuda=True, activation='relu'):
        BaseMeanFunction.__init__(self, seed=seed, number_of_functions=len(dataset))
        self._nn = ResNet18(activation_name=activation)
        self._is_cuda = is_cuda
        if self._is_cuda:
            self._nn = self._nn.cuda()
            cudnn.benchmark = True
        self._criterion = nn.CrossEntropyLoss()
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._batch_sampler = iter(torch.utils.data.DataLoader(
            self._dataset, batch_size=self._batch_size, num_workers=num_workers, 
            sampler=torch.utils.data.RandomSampler(self._dataset, replacement=True, num_samples=self._LARGE_NUMBER)))
        self._last_loss = None
        self._wd = 0.0
    
    def statistics(self):
        return {'last_loss': self._last_loss,
                'last_accuracy': self._last_accuracy}
    
    def stochastic_gradient_at_points(self, points):
        return self.batch_gradient_at_points(points, self._batch_size)
    
    def batch_gradient_at_points(self, points, batch_size):
        assert batch_size == self._batch_size, (batch_size, self._batch_size)
        features, labels = next(self._batch_sampler)
        if self._is_cuda:
            features, labels = features.cuda(), labels.cuda()
        gradients = []
        for point in points:
            gradients.append(self._gradient(point, features, labels))
        return gradients
    
    def gradient(self, point):
        aggregated_gradient = 0.0
        num_batches = 0
        for features, labels in torch.utils.data.DataLoader(self._dataset, self._batch_size, num_workers=self._num_workers):
            if self._is_cuda:
                features, labels = features.cuda(), labels.cuda()
            aggregated_gradient += self._gradient(point, features, labels)
            num_batches += 1
        return aggregated_gradient / num_batches
    
    def value(self, point):
        aggregated_value = 0.0
        num_batches = 0
        with torch.no_grad():
            for features, labels in torch.utils.data.DataLoader(self._dataset, self._batch_size, num_workers=self._num_workers):
                if self._is_cuda:
                    features, labels = features.cuda(), labels.cuda()
                tensor_to_parameters(self._nn.parameters(), point)
                logits = self._nn(features)
                loss = self._loss(logits, labels)
                aggregated_value += loss
                num_batches += 1
        return (aggregated_value / num_batches).detach()
    
    def dim(self):
        parameters_flatten = parameters_to_tensor(self._nn.parameters())
        return parameters_flatten.shape[0]
    
    def get_current_point(self):
        return parameters_to_tensor(self._nn.parameters())
    
    def _gradient(self, point, features, labels):
        self._nn.zero_grad()
        tensor_to_parameters(self._nn.parameters(), point)
        logits = self._nn(features)
        loss = self._loss(logits, labels)
        loss.backward()
        self._last_loss = loss.detach().cpu().numpy()
        self._last_accuracy = self._accuracy(logits, labels)
        return parameters_to_tensor(self._nn.parameters(), grad=True)

    def _loss(self, logits, labels):
        loss = self._criterion(logits, labels)
        if self._wd > 0:
            all_parameters = []
            for parameter in self._nn.parameters():
                all_parameters.append(parameter.flatten())
            all_parameters = torch.cat(all_parameters)
            loss = loss + (self._wd / 2.0) * torch.dot(all_parameters.reshape(-1), 
                                                       all_parameters.reshape(-1))
        return loss

    def _accuracy(self, logits, labels):
        prediction = np.argmax(logits.detach().cpu().numpy(), axis=1)
        labels_np = labels.detach().cpu().numpy()
        acc = np.sum(prediction == labels_np) / float(len(labels_np))
        return acc
    
    def _check_accuracy(self, point):
        accuracy = 0.0
        num = 0
        tensor_to_parameters(self._nn.parameters(), point)
        for features, labels in torch.utils.data.DataLoader(self._dataset, self._batch_size, num_workers=self._num_workers):
            if self._is_cuda:
                features, labels = features.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self._nn(features)
                accuracy += self._accuracy(logits, labels)
            num += 1
        return accuracy / num
    
    def type(self):
        if self._is_cuda:
            return FunctionType.TORCH_CUDA
        else:
            return FunctionType.TORCH_CPU


class BaseMLFunction(BaseFunction):
    def __init__(self, features, number_of_classes=2, binary=True, no_copy_features=False, **kwargs):
        super(BaseMLFunction, self).__init__(**kwargs)
        if no_copy_features:
            self._features_np = features
            self._features = torch.from_numpy(self._features_np)
        else:
            ones_features = np.ones((len(features), 1), dtype=np.float32)
            self._features_np = np.concatenate((features, ones_features), axis=1)
            self._features = torch.tensor(self._features_np)
        self._number_of_classes = number_of_classes
        self._binary = binary
    
    def dim(self):
        if self._binary:
            return self._features.shape[1]
        else:
            return self._features.shape[1] * self._number_of_classes
    
    def _preprocess_point(self, point):
        if not self._binary:
            point = np.reshape(point, [-1, self._number_of_classes])
        return point


class BaseGradientMLFunction(BaseMLFunction, BaseMeanFunction):
    def __init__(self, features, number_of_classes=2, binary=True, seed=None,
                 number_of_functions=None, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT,
                 no_copy_features=False):
        super(BaseGradientMLFunction, self).__init__(
            features, number_of_classes=number_of_classes, 
            binary=binary, seed=seed, number_of_functions=number_of_functions,
            no_copy_features=no_copy_features)
        self._sampling = sampling
        self._importance_probs = None
    
    def value(self, point):
        point = self._preprocess_point(point)
        point_torch = torch.tensor(point, requires_grad=True)
        return self._loss(point_torch).detach().numpy()
    
    def gradient(self, point):
        point = self._preprocess_point(point)
        point_torch = torch.tensor(point, requires_grad=True)
        loss = self._loss(point_torch)
        loss.backward()
        grad_numpy = point_torch.grad.detach().numpy()
        grad_numpy = grad_numpy.reshape(-1)
        return grad_numpy
    
    def batch_gradient_at_points(self, points, batch_size):
        if self._sampling == SamplingType.UNIFORM_WITH_REPLACEMENT:
            indices = self._generator.integers(self._number_of_functions, size=batch_size)
        if self._sampling == SamplingType.IMPORTANCE:
            assert batch_size == 1
            if self._importance_probs is None:
                liptschitz_local = self.liptschitz_local_gradient_constants()
                self._importance_probs = liptschitz_local / np.sum(liptschitz_local)
            indices = self._generator.choice(self._number_of_functions, size=batch_size,
                                             p=self._importance_probs)
        return self.batch_gradient_at_points_with_indices(points, indices)

    def batch_gradient_at_points_with_indices(self, points, indices):
        batch_features = self._features[indices, :]
        batch_labels = self._labels[indices]
        batch_gradients = []
        for point in points:
            batch_gradient = self._gradient_features_labels(point, batch_features, batch_labels)
            if self._sampling == SamplingType.IMPORTANCE:
                assert len(indices) == 1
                batch_gradient = batch_gradient / (self._number_of_functions * self._importance_probs[indices[0]])
            batch_gradients.append(batch_gradient)
        return batch_gradients

    def _gradient_features_labels(self, point, features, labels):
        point = self._preprocess_point(point)
        point_torch = torch.tensor(point, requires_grad=True)
        loss = self._loss_features_labels(point_torch, features, labels)
        loss.backward()
        grad_numpy = point_torch.grad.detach().numpy()
        grad_numpy = grad_numpy.reshape(-1)
        return grad_numpy
    
    def _loss(self, point):
        raise NotImplementedError()
    
    def _loss_features_labels(self, point):
        raise NotImplementedError()
    
    def liptschitz_local_gradient_constants(self):
        raise NotImplementedError()


class NonConvexLossFunction(BaseGradientMLFunction, BaseStochasticFunction):
    def __init__(self, features, labels, seed=None, check_accuracy=False, sampling=SamplingType.UNIFORM_WITH_REPLACEMENT,
                 batch_size=None):
        super(NonConvexLossFunction, self).__init__(features, seed=seed, number_of_functions=len(labels),
                                                    sampling=sampling)
        assert labels.dtype == np.int64
        assert np.unique(labels).tolist() == [0, 1], np.unique(labels).tolist()
        labels = 2 * labels - 1
        self._labels_np = labels
        self._labels = torch.tensor(self._labels_np)
        self._check_accuracy_flag = check_accuracy
        self._batch_size = batch_size
    
    def liptschitz_local_gradient_constants(self):
        bound_for_second_derivative = 0.15406
        square_norm = np.sum(np.square(self._features_np), axis=1)
        assert len(square_norm) == self._features_np.shape[0]
        return bound_for_second_derivative * square_norm
    
    def liptschitz_max_gradient_constant(self):
        liptschitz_local = self.liptschitz_local_gradient_constants()
        liptschitz_constant = np.max(liptschitz_local)
        return liptschitz_constant
    
    def liptschitz_plus_gradient_constant(self):
        liptschitz_local = self.liptschitz_local_gradient_constants()
        return np.sqrt(np.sum(liptschitz_local ** 2))
    
    def liptschitz_gradient_constant(self):
        return self.liptschitz_max_gradient_constant()
    
    def _check_accuracy(self, point):
        point = self._preprocess_point(point)
        point_torch = torch.tensor(point, requires_grad=True)
        proj = torch.mv(self._features, point_torch)
        prob = torch.sigmoid(proj)
        auc = roc_auc_score(self._labels_np, prob.detach().numpy())
        return auc
    
    @staticmethod
    def _loss_features_labels(point, features, labels):
        proj = torch.mv(features, point)
        proj_labels = proj * labels
        prob_proj = torch.sigmoid(proj_labels)
        loss = torch.square(1 - prob_proj)
        loss = torch.mean(loss)
        return loss
    
    def _loss(self, point):
        return self._loss_features_labels(point, self._features, self._labels)
    
    def stochastic_gradient_at_points(self, points):
        return self.batch_gradient_at_points(points, self._batch_size)


class NonConvexLossMultiClassFunction(BaseGradientMLFunction):
    def __init__(self, features, labels, check_accuracy=False, number_of_classes=2, reg_paramterer=1e-4,
                 seed=None):
        super(NonConvexLossMultiClassFunction, self).__init__(features, number_of_classes, binary=False,
                                                              seed=seed, number_of_functions=len(labels))
        self._reg_paramterer = reg_paramterer
        assert labels.dtype == np.int64
        unique_labels = set(np.unique(labels).tolist())
        expected_classes = set(range(0, number_of_classes))
        assert unique_labels <= expected_classes, unique_labels
        if unique_labels != expected_classes:
            print('Warning: Dataset has only the partial number of classes')
        self._labels_np = labels
        self._labels = torch.tensor(self._labels_np)
        self._labels_one_hot = torch.nn.functional.one_hot(self._labels, 
                                                           num_classes=number_of_classes).float()
        self._input_dim = features.shape[1]
        self._check_accuracy_flag = check_accuracy
        self._softmax = torch.nn.Softmax(dim=1)
        
    def _loss(self, point):
        proj = torch.mm(self._features, point)
        prob_proj = self._softmax(proj)
        prob_proj_labels = prob_proj * self._labels_one_hot
        prob_proj_labels = torch.sum(prob_proj_labels, axis=1)
        loss = torch.square(1 - prob_proj_labels)
        loss = torch.mean(loss)
        loss = loss + (self._reg_paramterer / 2.0) * torch.dot(point.reshape(-1), 
                                                               point.reshape(-1))
        if self._check_accuracy_flag:
            self._check_accuracy(proj.detach())
        return loss
    
    def _check_accuracy(self, point):
        point = self._preprocess_point(point)
        point = torch.tensor(point, requires_grad=True)
        proj = torch.mm(self._features, point)
        prob = self._softmax(proj)
        prediction = np.argmax(prob.detach().numpy(), axis=1)
        acc = np.sum(prediction == self._labels_np) / float(len(self._labels_np))
        return acc
    
    def liptschitz_gradient_constant(self):
        bound_for_second_derivative = 12 * self._number_of_classes  # very raw estimation
        square_norm = np.sum(np.square(self._features_np), axis=1)
        assert len(square_norm) == self._features_np.shape[0]
        liptschitz_constant = bound_for_second_derivative * np.max(square_norm)
        return liptschitz_constant
    
    def deterministic(self):
        return True


class LogisticRegressionFunction(BaseGradientMLFunction, BaseStochasticFunction):
    def __init__(self, features, labels, check_accuracy=False, number_of_classes=2, 
                 reg_paramterer=0.0, nonconvex_regularizer=False, seed=None,
                 sampling=SamplingType.UNIFORM_WITH_REPLACEMENT, batch_size=None,
                 no_copy_features=False):
        super(LogisticRegressionFunction, self).__init__(features, number_of_classes, binary=False,
                                                         seed=seed, number_of_functions=len(labels),
                                                         sampling=sampling, no_copy_features=no_copy_features)
        self._log_loss = torch.nn.CrossEntropyLoss()
        self._reg_paramterer = reg_paramterer
        assert labels.dtype == np.int64
        unique_labels = set(np.unique(labels).tolist())
        expected_classes = set(range(0, number_of_classes))
        assert unique_labels <= expected_classes, unique_labels
        if unique_labels != expected_classes:
            print('Warning: Dataset has only the partial number of classes: number of samples: {}, unique classes: {}'.\
                format(len(labels), unique_labels))
        self._labels_np = labels
        self._labels = torch.tensor(self._labels_np)
        self._input_dim = features.shape[1]
        self._check_accuracy_flag = check_accuracy
        self._nonconvex_regularizer = nonconvex_regularizer
        self._batch_size = batch_size

    def _loss(self, point):
        return self._loss_features_labels(point, self._features, self._labels)
    
    def _loss_features_labels(self, point, features, labels):
        proj = torch.mm(features, point)
        loss = self._log_loss(proj, labels)
        if self._reg_paramterer > 0.0:
            if not self._nonconvex_regularizer:
                reg = (self._reg_paramterer / 2.0) * torch.dot(point.reshape(-1), point.reshape(-1))
            else:
                square_point = torch.square(point)
                reg = self._reg_paramterer * torch.sum(square_point / (1. + square_point))
            loss = loss + reg
        return loss
    
    def _check_accuracy(self, point):
        point = self._preprocess_point(point)
        point = torch.tensor(point, requires_grad=True)
        proj = torch.mm(self._features, point)
        prob = torch.softmax(proj, 1)
        prediction = np.argmax(prob.detach().numpy(), axis=1)
        acc = np.sum(prediction == self._labels_np) / float(len(self._labels_np))
        return acc
    
    def liptschitz_local_gradient_constants(self):
        bound_for_second_derivative = 0.25
        square_norm = np.sum(np.square(self._features_np), axis=1)
        assert len(square_norm) == self._features_np.shape[0]
        liptschitz_local = bound_for_second_derivative * square_norm
        if self._reg_paramterer > 0.0:
            if not self._nonconvex_regularizer:
                liptschitz_local += self._reg_paramterer
            else:
                liptschitz_local += 2 * self._reg_paramterer
        return liptschitz_local

    def stochastic_gradient_at_points(self, points):
        return self.batch_gradient_at_points(points, self._batch_size)


class LogisticRegressionOptimizationProblemMeta(OptimizationProblemMeta):
    def __init__(self, functions):
        self._functions = functions
    
    def is_convex(self):
        return all([not f._nonconvex_regularizer for f in self._functions])
    
    def strongly_convex_constant(self):
        assert self.is_convex()
        return sum([f._reg_paramterer for f in self._functions]) / float(len(self._functions))


class StochasticLogisticRegressionFunction(LogisticRegressionFunction, BaseStochasticFunction):
    def __init__(self, *args, **kwargs):
        super(StochasticLogisticRegressionFunction, self).__init__(*args, **kwargs)
