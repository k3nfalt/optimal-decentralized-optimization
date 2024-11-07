import numpy as np
import pytest
import itertools

from distributed_optimization_library.function import StochasticQuadraticFunction
from distributed_optimization_library.function import generate_random_vector
from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.asynchronous.algorithm_with_graphs import MiniBatchSGD, MiniBatchSGDNode, FragileSGD, add_torus
from distributed_optimization_library.signature import Signature


def mean(vectors):
    return sum(vectors) / float(len(vectors))

@pytest.mark.parametrize("type_graph,method", 
                         list(itertools.product(["line", "torus"], ["minibatch", "fragile"])))
def test_minibatch_sgd_with_quadratic_function_and_comm(type_graph, method):
    time_transfer = 1
    gamma = 0.01
    dim = 100
    num_iterations = 10000
    
    generator = np.random.default_rng(seed=42)
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    time_to_calulate_gradient = 100
    if type_graph == "line":
        num_nodes = 11
        delays = {"calculate_stochastic_gradient": [time_to_calulate_gradient] * num_nodes}
        for to_node in range(num_nodes):
            delays[f"send_vector_to_{to_node}"] = [np.inf] * num_nodes
            delays[f"send_vector_to_{to_node}"][to_node] = 0
            if to_node - 1 >= 0:
                delays[f"send_vector_to_{to_node}"][to_node - 1] = time_transfer
            if to_node + 1 < num_nodes:
                delays[f"send_vector_to_{to_node}"][to_node + 1] = time_transfer
    elif type_graph == "torus":
        num_nodes = 11 ** 2
        delays = {"calculate_stochastic_gradient": [time_to_calulate_gradient] * num_nodes}
        add_torus(delays, num_nodes, distance=time_transfer)
    nodes = [Signature(MiniBatchSGDNode, function, num_nodes, id=index) 
             for index in range(num_nodes)]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    if method == "minibatch":
        optimizer = MiniBatchSGD(transport, point, gamma, gamma_multiply=None)
    elif method == "fragile":
        optimizer = FragileSGD(transport, point, gamma, 
                               gamma_multiply=None, batch_size=30)
    for index in range(num_iterations):
        optimizer.step()
        if type_graph == "line":
            correct_time = (((num_nodes - 1) // 2) * dim * time_transfer + 
                            time_to_calulate_gradient + 
                            ((num_nodes - 1) // 2) * dim * time_transfer) * (index + 1)
            if method == "minibatch":
                assert optimizer.get_time() == correct_time
            elif method == "fragile":
                assert optimizer.get_time() <= correct_time
        elif type_graph == "torus":
            num_nodes_on_edge = int(np.sqrt(num_nodes))
            correct_time = (num_nodes_on_edge * dim * time_transfer
                            + time_to_calulate_gradient 
                            + num_nodes_on_edge * dim * time_transfer) * (index + 1)
            if method == "minibatch":
                assert optimizer.get_time() >= correct_time and optimizer.get_time() <= 2 * correct_time
            elif method == "fragile":
                assert optimizer.get_time() <= correct_time
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)


@pytest.mark.parametrize("type_graph,method", list(itertools.product(["line"], ["minibatch", "fragile"])))
def test_minibatch_sgd_with_quadratic_function_and_free_comm(type_graph, method):
    time_transfer = 0
    gamma = 0.01
    dim = 100
    num_iterations = 10000
    
    generator = np.random.default_rng(seed=42)
    function = StochasticQuadraticFunction.create_random(dim, generator, noise=0.1, reg=0.1)
    time_to_calulate_gradient = 100
    if type_graph == "line":
        num_nodes = 11
        delays = {"calculate_stochastic_gradient": [time_to_calulate_gradient] * num_nodes}
        for to_node in range(num_nodes):
            delays[f"send_vector_to_{to_node}"] = [np.inf] * num_nodes
            delays[f"send_vector_to_{to_node}"][to_node] = 0
            if to_node - 1 >= 0:
                delays[f"send_vector_to_{to_node}"][to_node - 1] = time_transfer
            if to_node + 1 < num_nodes:
                delays[f"send_vector_to_{to_node}"][to_node + 1] = time_transfer
    elif type_graph == "torus":
        num_nodes = 11 ** 2
        delays = {"calculate_stochastic_gradient": [time_to_calulate_gradient] * num_nodes}
        add_torus(delays, num_nodes, distance=time_transfer)
    nodes = [Signature(MiniBatchSGDNode, function, num_nodes, id=index) 
             for index in range(num_nodes)]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    point = generate_random_vector(dim, generator)
    if method == "minibatch":
        optimizer = MiniBatchSGD(transport, point, gamma, gamma_multiply=None)
    elif method == "fragile":
        optimizer = FragileSGD(transport, point, gamma, 
                               gamma_multiply=None, batch_size=30)
    for index in range(num_iterations):
        optimizer.step()
        if type_graph == "line":
            if method == "minibatch":
                correct_time = (time_to_calulate_gradient) * (index + 1)
                assert optimizer.get_time() == correct_time
        elif type_graph == "torus":
            num_nodes_on_edge = int(np.sqrt(num_nodes))
            if method == "minibatch":
                correct_time = (num_nodes_on_edge * dim * time_transfer
                                + time_to_calulate_gradient 
                                + num_nodes_on_edge * dim * time_transfer) * (index + 1)
                assert optimizer.get_time() >= correct_time and optimizer.get_time() <= 2 * correct_time
    solution = optimizer.get_point()
    
    analytical_solution = np.linalg.solve(function._quadratic_function._A, function._quadratic_function._b)
    np.testing.assert_array_almost_equal(solution, analytical_solution, decimal=2)