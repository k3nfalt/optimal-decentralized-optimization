import pytest

import numpy as np

from distributed_optimization_library.function import NeuralNetworkFunction, NonConvexLossFunction, \
    generate_random_vector, LogisticRegressionFunction, TridiagonalQuadraticFunction, QuadraticFunction, \
    NonConvexLossMultiClassFunction, StochasticLogisticRegressionFunction


def test_neural_network_smoke_test():
    num_samples = 13
    features = np.random.randn(num_samples, 23).astype(np.float32)
    labels = np.random.randint(2, size=(num_samples,))
    func = NeuralNetworkFunction(features, labels, neural_network_name='two_layer_neural_net')
    point = func.get_current_point()
    value_at_point = func.value(point)
    gradient_at_point = func.gradient(point)
    func.value(np.zeros_like(point))
    func.gradient(np.zeros_like(point))
    np.testing.assert_array_almost_equal(gradient_at_point, func.gradient(point))
    np.testing.assert_array_almost_equal(value_at_point, func.value(point))


def test_non_convex_function_smoke_test():
    num_samples = 13
    features = np.random.randn(num_samples, 23).astype(np.float32)
    labels = np.random.randint(2, size=(num_samples,))
    func = NonConvexLossFunction(features, labels)
    point = generate_random_vector(24, 42).astype(np.float32)
    value_at_point = func.value(point)
    gradient_at_point = func.gradient(point)
    func.value(np.zeros_like(point))
    func.gradient(np.zeros_like(point))
    np.testing.assert_array_almost_equal(gradient_at_point, func.gradient(point))
    np.testing.assert_array_almost_equal(value_at_point, func.value(point))
    
    
def test_non_convex_function_multiclass_smoke_test():
    num_samples = 13
    features = np.random.randn(num_samples, 23).astype(np.float32)
    labels = np.random.randint(4, size=(num_samples,))
    func = NonConvexLossMultiClassFunction(features, labels, number_of_classes=4)
    point = generate_random_vector(24 * 4, 42).astype(np.float32)
    value_at_point = func.value(point)
    gradient_at_point = func.gradient(point)
    func.value(np.zeros_like(point))
    func.gradient(np.zeros_like(point))
    np.testing.assert_array_almost_equal(gradient_at_point, func.gradient(point))
    np.testing.assert_array_almost_equal(value_at_point, func.value(point))
    

def test_batch_function_smoke_test():
    num_samples = 13
    features = np.random.randn(num_samples, 23).astype(np.float32)
    labels = np.random.randint(2, size=(num_samples,))
    for func in [NonConvexLossFunction, LogisticRegressionFunction]:
        mean_func = func(features, labels, seed=42)
        if func == NonConvexLossFunction:
            point = generate_random_vector(dim=24, seed=42).astype(np.float32)
        else:
            point = generate_random_vector(dim=48, seed=42).astype(np.float32)
        batch_gradient = 0.0
        num_samples = 2000
        for _ in range(num_samples):
            batch_gradient += mean_func.batch_gradient(point, batch_size=4)
        batch_gradient /= num_samples
        mean_gradient = mean_func.gradient(point)
        
        func = func(features, labels)
        gradient = func.gradient(point)
        
        np.testing.assert_array_almost_equal(batch_gradient, gradient, decimal=1)
        np.testing.assert_array_almost_equal(mean_gradient, gradient)


def test_stochastic_function_smoke_test():
    num_samples = 13
    features = np.random.randn(num_samples, 23).astype(np.float32)
    labels = np.random.randint(2, size=(num_samples,))
    func = StochasticLogisticRegressionFunction
    mean_func = func(features=features, labels=labels, seed=42, batch_size=12)
    point = generate_random_vector(dim=48, seed=42).astype(np.float32)
    batch_gradient = 0.0
    num_samples = 2000
    for _ in range(num_samples):
        batch_gradient += mean_func.stochastic_gradient(point)
    batch_gradient /= num_samples
    mean_gradient = mean_func.gradient(point)
    
    func = func(features=features, labels=labels, seed=42, batch_size=12)
    gradient = func.gradient(point)
    
    np.testing.assert_array_almost_equal(batch_gradient, gradient, decimal=1)
    np.testing.assert_array_almost_equal(mean_gradient, gradient)


def test_logistic_regression_function_smoke_test():
    num_samples = 13
    features = np.random.randn(num_samples, 23).astype(np.float32)
    labels = np.random.randint(2, size=(num_samples,))
    func = LogisticRegressionFunction(features, labels)
    point = generate_random_vector(dim=48, seed=42).astype(np.float32)
    value_at_point = func.value(point)
    gradient_at_point = func.gradient(point)
    assert gradient_at_point.ndim == 1
    func.value(np.zeros_like(point))
    func.gradient(np.zeros_like(point))
    np.testing.assert_array_almost_equal(gradient_at_point, func.gradient(point))
    np.testing.assert_array_almost_equal(value_at_point, func.value(point))


def test_tridiagonal_function():
    tridiag_quad = TridiagonalQuadraticFunction.create_worst_case_functions(1, 100)[0]
    quad = QuadraticFunction(tridiag_quad._A.toarray(), tridiag_quad._b)
    point = generate_random_vector(100, 42)
    np.testing.assert_almost_equal(tridiag_quad.value(point), quad.value(point))
    np.testing.assert_almost_equal(tridiag_quad.gradient(point), quad.gradient(point))
    

def test_tridiagonal_function_stats():
    tridiag_quads = TridiagonalQuadraticFunction.create_worst_case_functions(10, 100, noise_lambda=1.0)
    quads = [QuadraticFunction(tridiag_quad._A.toarray(), tridiag_quad._b) 
             for tridiag_quad in tridiag_quads]
    np.testing.assert_almost_equal(
        TridiagonalQuadraticFunction.smoothness_variance_bound_functions(tridiag_quads),
        QuadraticFunction.smoothness_variance_bound_functions(quads), decimal=4)
    np.testing.assert_almost_equal(
        TridiagonalQuadraticFunction.liptschitz_gradient_constant_functions(tridiag_quads),
        QuadraticFunction.liptschitz_gradient_constant_functions(quads), decimal=5)
    np.testing.assert_almost_equal(
        TridiagonalQuadraticFunction.min_eigenvalue_functions(tridiag_quads),
        QuadraticFunction.min_eigenvalue_functions(quads), decimal=5)
    np.testing.assert_almost_equal(
        TridiagonalQuadraticFunction.analytical_solution_functions(tridiag_quads) / 10.,
        QuadraticFunction.analytical_solution_functions(quads) / 10., decimal=1)
