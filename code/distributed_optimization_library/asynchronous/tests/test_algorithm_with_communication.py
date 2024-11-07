import numpy as np

from distributed_optimization_library.function import StochasticQuadraticFunction
from distributed_optimization_library.function import generate_random_vector
from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.asynchronous.algorithm_with_communication import MiniBatchSGD, MiniBatchSGDNode, \
    FastestSGDNode, FastestSGD, QSGD, QSGDNode, AsynchronousSGD, AsynchronousSGDNode, find_equilibrium_time, Shadowheart, ShadowheartNode
from distributed_optimization_library.asynchronous.algorithm_with_communication import METHOD_SEND, METHOD_GRADIENT
from distributed_optimization_library.signature import Signature


def mean(vectors):
    return sum(vectors) / float(len(vectors))


def test_minibatch_sgd_with_quadratic_function():
    gamma = 0.01
    dim = 100
    num_iterations = 10000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(MiniBatchSGDNode, function) for _ in range(num_nodes)]
    delays = {METHOD_GRADIENT: [(i + 1) for i in range(num_nodes)],
              METHOD_SEND: [(i + 1) / dim for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = MiniBatchSGD(transport, point, gamma, gamma_multiply=None)
    for index in range(num_iterations):
        optimizer.step()
        assert optimizer.get_time() == 2 * num_nodes * (index + 1)
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_fastest_sgd_with_quadratic_function():
    gamma = 0.01
    dim = 100
    num_iterations = 100000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(FastestSGDNode, function) for _ in range(num_nodes)]
    delays = {METHOD_GRADIENT: [(i + 1) for i in range(num_nodes)],
              METHOD_SEND: [(i + 1) / dim for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = FastestSGD(transport, point, gamma, gamma_multiply=None)
    for index in range(num_iterations):
        optimizer.step()
        assert optimizer.get_time() == 1 * (index + 1)
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_qsgd_with_quadratic_function():
    gamma = 0.005
    dim = 100
    num_iterations = 50000
    number_of_coordinates = 10
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(QSGDNode, function) for _ in range(num_nodes)]
    delays = {METHOD_GRADIENT: [(i + 1) for i in range(num_nodes)],
              METHOD_SEND: [(i + 1) / dim for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = QSGD(transport, point, gamma, gamma_multiply=None, number_of_coordinates=number_of_coordinates)
    for index in range(num_iterations):
        optimizer.step()
        np.testing.assert_array_almost_equal(optimizer.get_time(), (10 + 10 * number_of_coordinates / dim) * (index + 1))
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=1)


def test_asynchronous_sgd_with_quadratic_function():
    gamma = 0.001
    dim = 100
    num_iterations = 300000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(AsynchronousSGDNode, function) for _ in range(num_nodes)]
    delays = {METHOD_GRADIENT: [(i + 1) for i in range(num_nodes)],
              METHOD_SEND: [(i + 1) / dim for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = AsynchronousSGD(transport, point, gamma)
    for _ in range(num_iterations):
        optimizer.step()
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


def test_shadowheart_sgd_with_quadratic_function():
    gamma = 0.01
    dim = 100
    num_iterations = 10000
    
    generator = np.random.default_rng(seed=42)
    num_nodes = 10
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    nodes = [Signature(ShadowheartNode, function) for _ in range(num_nodes)]
    ratio = 5
    delays = {METHOD_GRADIENT: [(i + 1) for i in range(num_nodes)],
              METHOD_SEND: [(i + 1) / ratio for i in range(num_nodes)]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    optimizer = Shadowheart(transport, point, gamma, number_of_coordinates=10, sigma_eps=1)
    time = 0
    for _ in range(num_iterations):
        optimizer.step()
        time += 2 * optimizer._equilibrium_time
        assert optimizer.get_time() <= time
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)
