import copy

from enum import Enum


class State(Enum):
    AVAILABLE = 1
    WORKING = 2
    
    
class ReturnState(Enum):
    WAIT = 1
    RESULT = 2


class MethodDelayedAsynchronousTransport(object):
    def __init__(self, nodes, delays):
        self._nodes = [node.create_instance() for node in nodes]
        if isinstance(delays, (list, tuple)):
            DEFAULT_METHOD = 'calculate_stochastic_gradient'
            delays = {DEFAULT_METHOD: delays}
        self._delays = delays
        self._methods = delays.keys()
        self._current_time = 0
        self._reset_all_nodes()
    
    def call_available_node_method(self, time, node_index, node_method, force_lazy=False, **kwargs):
        assert self._current_time <= time
        self._current_time = time
        assert self._states[node_method][node_index] == State.AVAILABLE
        self._states[node_method][node_index] = State.WORKING
        cost_method = "cost_" + node_method
        cost = getattr(self._nodes[node_index], cost_method)(**kwargs)
        self._time_return[node_method][node_index] = self._current_time + cost * self._delays[node_method][node_index]
        if not force_lazy:
            self._outputs[node_method][node_index] = (force_lazy, getattr(self._nodes[node_index], node_method)(**kwargs))
        else:
            self._outputs[node_method][node_index] = (force_lazy, [node_index, node_method, copy.deepcopy(kwargs)])
        return self._time_return[node_method][node_index]
    
    def call_ready_node(self, time, node_index, node_method):
        assert self._current_time <= time
        assert self._states[node_method][node_index] == State.WORKING
        self._current_time = time
        assert self._current_time >= self._time_return[node_method][node_index]
        self._states[node_method][node_index] = State.AVAILABLE
        force_lazy, output = self._outputs[node_method][node_index]
        if force_lazy:
            node_index_, node_method_, kwargs_ = output
            output = getattr(self._nodes[node_index_], node_method_)(**kwargs_)
        return output
    
    def call_node_method(self, node_index, node_method, **kwargs):
        assert node_method not in self._methods
        return getattr(self._nodes[node_index], node_method)(**kwargs)
    
    def call_nodes_method(self, node_method, **kwargs):
        assert node_method not in self._methods
        return [self.call_node_method(node_index, node_method, **kwargs)
                for node_index in range(self.get_number_of_nodes())]
    
    def get_number_of_nodes(self):
        return len(self._nodes)
    
    def get_methods(self):
        return self._methods
    
    def get_delays(self):
        return self._delays
    
    def get_time(self):
        return self._current_time
    
    def stop_all_calculations(self):
        self._reset_all_nodes()

    def _reset_all_nodes(self):
        self._states = {k: [State.AVAILABLE] * len(self._nodes) for k in self._methods}
        self._outputs = {k: [None] * len(self._nodes) for k in self._methods}
        self._time_return = {k: [None] * len(self._nodes) for k in self._methods}
