from distributed_optimization_library.asynchronous.asynchronous_transport import MethodDelayedAsynchronousTransport
from distributed_optimization_library.signature import Signature


class DummyNode(object):
    def __init__(self, value):
        self._value = value
    
    def cost_value(self):
        return 1
    
    def value(self):
        return self._value
    
    def cost_gradient(self):
        return 1
    
    def gradient(self):
        return self._value


def test_one_node_success_run():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value")
    output = transport.call_ready_node(time, 0, "value")
    assert output == 0


def test_one_node_success_run():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value", force_lazy=True)
    output = transport.call_ready_node(time, 0, "value")
    assert output == 0


def test_one_node_success_run_with_the_methods():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10], "gradient": [2, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value")
    time_gradient = transport.call_available_node_method(time, 0, "gradient")
    output = transport.call_ready_node(time, 0, "value")
    output_gradient = transport.call_ready_node(time_gradient, 0, "gradient")
    assert output == 0
    assert output_gradient == 0


def test_one_node_fail_run_with_the_methods_other_order():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10], "gradient": [2, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value")
    time_gradient = transport.call_available_node_method(time, 0, "gradient")
    output_gradient = transport.call_ready_node(time_gradient, 0, "gradient")
    try:
        output = transport.call_ready_node(time, 0, "value")
    except AssertionError:
        return
    assert False


def test_one_node_success_run_with_the_methods_other_order():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10], "gradient": [2, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value")
    time_gradient = transport.call_available_node_method(time, 0, "gradient")
    output_gradient = transport.call_ready_node(time_gradient, 0, "gradient")
    output = transport.call_ready_node(time_gradient, 0, "value")
    assert output == 0
    assert output_gradient == 0


def test_one_node_fail_run_too_early():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value") - 0.5
    try:
        output = transport.call_ready_node(time, 0, "value")
    except AssertionError:
        return
    assert False


def test_one_node_fail_run_double_call_available():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value")
    try:
        transport.call_available_node_method(time, 0, "value")
    except AssertionError:
        return
    assert False


def test_one_node_fail_run_double_call_ready():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    time = transport.call_available_node_method(time, 0, "value")
    transport.call_ready_node(time, 0, "value")
    try:
        transport.call_ready_node(time, 0, "value")
    except AssertionError:
        return
    assert False


def test_one_node_fail_run_call_ready_first():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    try:
        transport.call_ready_node(time, 0, "value")
    except AssertionError:
        return
    assert False


def test_two_nodes_success_run():
    nodes = [Signature(DummyNode, 0), Signature(DummyNode, 1)]
    delays = {"value": [1, 10]}
    transport = MethodDelayedAsynchronousTransport(nodes, delays)
    time = 0
    transport.call_available_node_method(time, 0, "value")
    transport.call_available_node_method(time, 1, "value")
    assert transport.call_ready_node(time + 2, 0, "value") == 0
    assert transport.call_ready_node(time + 10, 1, "value") == 1
