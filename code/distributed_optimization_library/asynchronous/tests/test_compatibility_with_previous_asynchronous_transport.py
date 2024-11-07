from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.signature import Signature


class DummyNode(object):
    def __init__(self, value):
        self._value = value
    
    def cost_calculate_stochastic_gradient(self):
        return 1.
    
    def calculate_stochastic_gradient(self):
        return self._value
    
    def free_calculate_stochastic_gradient(self):
        return self._value


def test_one_node_success_run():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "calculate_stochastic_gradient")
    output = transport.call_ready_node(time, 0, "calculate_stochastic_gradient")
    assert output == 0


def test_one_node_fail_run_too_early():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "calculate_stochastic_gradient") - 0.5
    try:
        output = transport.call_ready_node(time, 0, "calculate_stochastic_gradient")
    except AssertionError:
        return
    assert False


def test_one_node_fail_run_double_call_available():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "calculate_stochastic_gradient")
    try:
        transport.call_available_node_method(time, 0, "calculate_stochastic_gradient")
    except AssertionError:
        return
    assert False


def test_one_node_fail_run_double_call_ready():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "calculate_stochastic_gradient")
    transport.call_ready_node(time, 0, "calculate_stochastic_gradient")
    try:
        transport.call_ready_node(time, 0, "calculate_stochastic_gradient")
    except AssertionError:
        return
    assert False


def test_one_node_fail_run_call_ready_first():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    try:
        transport.call_ready_node(time, 0, "calculate_stochastic_gradient")
    except AssertionError:
        return
    assert False


def test_two_nodes_success_run():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    transport.call_available_node_method(time, 0, "calculate_stochastic_gradient")
    transport.call_available_node_method(time, 1, "calculate_stochastic_gradient")
    assert transport.call_ready_node(time + 2, 0, "calculate_stochastic_gradient") == 0
    assert transport.call_ready_node(time + 10, 1, "calculate_stochastic_gradient") == 1


def test_call_nodes_method():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = [1, 10]
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    output = transport.call_nodes_method("free_calculate_stochastic_gradient")
    assert output == [0, 1]
