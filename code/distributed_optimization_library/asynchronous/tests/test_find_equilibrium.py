import numpy as np

from distributed_optimization_library.asynchronous.algorithm_with_communication import find_equilibrium_time


def test_find_equilibrium_one_node():
    number_of_coordinates = 1
    dim = 10
    sigma_eps = 3
    computation_times = np.array([4])
    communication_times = np.array([1])
    t = find_equilibrium_time(computation_times, communication_times, 
                              number_of_coordinates, dim, sigma_eps)
    assert (t > 0) and (t < np.inf)


def test_find_equilibrium_time_zero():
    number_of_coordinates = 1
    dim = 10
    sigma_eps = 3
    computation_times = np.array([4, 0, 5])
    communication_times = np.array([1, 0, 3])
    t = find_equilibrium_time(computation_times, communication_times, 
                              number_of_coordinates, dim, sigma_eps)
    np.testing.assert_equal(t, 0.0)


def test_find_equilibrium_time_inf():
    number_of_coordinates = 1
    dim = 10
    sigma_eps = 3
    computation_times = np.array([np.inf, np.inf, np.inf])
    communication_times = np.array([np.inf, np.inf, np.inf])
    t = find_equilibrium_time(computation_times, communication_times, 
                              number_of_coordinates, dim, sigma_eps)
    np.testing.assert_equal(t, np.inf)


def test_find_equilibrium_time_ignore_slow():
    number_of_coordinates = 1
    dim = 10
    sigma_eps = 3
    computation_times = np.array([4, 0, np.inf])
    communication_times = np.array([1, np.inf, 3])
    t = find_equilibrium_time(computation_times, communication_times, 
                              number_of_coordinates, dim, sigma_eps)
    assert (t > 0) and (t < np.inf)
    
    computation_times = np.array([4, 0, 10**2])
    communication_times = np.array([1, 1, 3])
    t = find_equilibrium_time(computation_times, communication_times, 
                              number_of_coordinates, dim, sigma_eps)
    assert (t > 0) and (t < 100)


def test_find_equilibrium_time_equal():
    number_of_coordinates = 1
    dim = 10
    sigma_eps = 3
    for dim in [1, 10, 100, 1000]:
        for sigma_eps in [0, 10, 100, 1000]:
            for number_of_coordinates in [1, 10, 20, 50, 100]:
                number_of_coordinates = min(number_of_coordinates, dim)
                for h in [0, 1, 10, 100, 1000]:
                    for tau in [0, 1, 10, 100, 1000]:
                        computation_times = np.array([h, h, h])
                        communication_times = np.array([tau, tau, tau])
                        t = find_equilibrium_time(computation_times, communication_times, 
                                                number_of_coordinates, dim, sigma_eps)
                        n = len(computation_times)
                        omega = dim / number_of_coordinates - 1
                        total_tau = number_of_coordinates * tau
                        exact_t = ((total_tau * omega / n + h * sigma_eps / n) + 
                                   np.sqrt((total_tau * omega / n + h * sigma_eps / n) ** 2 + 
                                           (4 * total_tau * h * sigma_eps * omega) / n))
                        exact_t = max(exact_t, max(h, total_tau))
                        np.testing.assert_almost_equal(t, exact_t)


def test_find_equilibrium_order_invariant():
    number_of_coordinates = 1
    dim = 10
    sigma_eps = 3
    n = 10
    for _ in range(100):
        computation_times = np.random.rand(n)
        communication_times = np.random.rand(n)
        results = []
        for _ in range(10):
            perm = np.random.permutation(n)
            computation_times = computation_times[perm]
            communication_times = communication_times[perm]
            t = find_equilibrium_time(computation_times, communication_times, 
                                    number_of_coordinates, dim, sigma_eps)
            results.append(t)
        np.testing.assert_almost_equal(results[1:], results[:-1])
