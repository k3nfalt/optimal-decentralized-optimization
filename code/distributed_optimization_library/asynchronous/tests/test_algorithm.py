import numpy as np

from distributed_optimization_library.function import StochasticQuadraticFunction, MeanQuadraticFunction
from distributed_optimization_library.function import generate_random_vector
from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.asynchronous.algorithm import StochasticGradientNodeAlgorithm, AsynchronousSGD, \
    AsynchronousMiniBatchSGD, MiniBatchSGD, AsynchronousPageMiniBatchSGD, MiniBatchPAGE
from distributed_optimization_library.signature import Signature


def mean(vectors):
    return sum(vectors) / float(len(vectors))


def test_asynchronous_sgd_with_quadratic_function():
    gamma = 0.001
    dim = 100
    num_iterations = 100000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    delays = [(i + 1) for i in range(num_nodes)]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = AsynchronousSGD(transport, point, gamma)
    for _ in range(num_iterations):
        optimizer.step()
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_asynchronous_minibatch_sgd_with_quadratic_function():
    gamma = 0.01
    batch_size = 10
    dim = 100
    num_iterations = 100000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    delays = [(i + 1) for i in range(num_nodes)]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = AsynchronousMiniBatchSGD(transport, point, gamma, gamma_multiply=None, batch_size=batch_size)
    for _ in range(num_iterations):
        optimizer.step()
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_minibatch_sgd_with_quadratic_function():
    gamma = 0.01
    batch_size = 10
    dim = 100
    num_iterations = 10000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    delays = [(i + 1) for i in range(num_nodes)]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = MiniBatchSGD(transport, point, gamma, gamma_multiply=None, batch_size=batch_size)
    for index in range(num_iterations):
        optimizer.step()
        assert optimizer.get_time() == (index + 1) * np.max(delays)
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_rennala_page_with_quadratic_function():
    gamma = 0.01
    number_of_functions = 10000
    batch_size = np.sqrt(number_of_functions)
    dim = 100
    num_nodes = 10
    num_iterations = 1000
    
    generator = np.random.default_rng(seed=42)
    function = MeanQuadraticFunction.create_random(dim, seed=generator, number_of_functions=number_of_functions, reg=0.1)
    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    delays = {'calculate_batch_gradient_at_points_with_indices': [(i + 1) for i in range(num_nodes)],
              'calculate_batch_gradient_at_points': [(i + 1) for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = AsynchronousPageMiniBatchSGD(transport, point, gamma, gamma_multiply=None, batch_size=batch_size)
    for index in range(num_iterations):
        optimizer.step()
    solution = optimizer.get_point()
    A = mean([qf._A for qf in function._quadratic_functions])
    B = mean([qf._b for qf in function._quadratic_functions])
    analytical_solution = np.linalg.solve(A, B)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_minibatch_page_with_quadratic_function():
    gamma = 0.01
    number_of_functions = 10000
    batch_size = None
    dim = 100
    num_nodes = 10
    num_iterations = 1000
    
    generator = np.random.default_rng(seed=42)
    function = MeanQuadraticFunction.create_random(dim, seed=generator, number_of_functions=number_of_functions, reg=0.1)
    nodes = [Signature(StochasticGradientNodeAlgorithm, function) for _ in range(num_nodes)]
    delays = {'calculate_batch_gradient_at_points_with_indices': [(i + 1) for i in range(num_nodes)],
              'calculate_batch_gradient_at_points': [(i + 1) for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = MiniBatchPAGE(transport, point, gamma, gamma_multiply=None, batch_size=batch_size)
    for index in range(num_iterations):
        optimizer.step()
    solution = optimizer.get_point()
    A = mean([qf._A for qf in function._quadratic_functions])
    B = mean([qf._b for qf in function._quadratic_functions])
    analytical_solution = np.linalg.solve(A, B)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)
