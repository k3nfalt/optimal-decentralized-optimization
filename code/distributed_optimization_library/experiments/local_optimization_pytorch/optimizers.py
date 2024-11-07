import torch
import numpy as np


class BaseOptimizer(object):
    def get_stat(self):
        return {}


class GradientDescent(BaseOptimizer):
    def __init__(self, model, point, lr):
        self._model = model
        self._point = point
        self._learning_rate = lr
    
    def step(self):
        gradient = self._model.gradient(self._point)
        self._point = self._point - self._learning_rate * gradient
    
    def get_point(self):
        return self._point


class SignGradientDescent(BaseOptimizer):
    def __init__(self, model, point, lr):
        self._model = model
        self._point = point
        self._learning_rate = lr
    
    def step(self):
        gradient = self._model.gradient(self._point)
        sign_gradient = (2 * (gradient > 0).float()) - 1
        self._point = self._point - self._learning_rate * sign_gradient
    
    def get_point(self):
        return self._point


class HolderGradientDescent(BaseOptimizer):
    def __init__(self, model, point, lr, nu=1.0):
        self._model = model
        self._point = point
        self._learning_rate = lr
        self._nu = nu
    
    def step(self):
        gradient = self._model.gradient(self._point)
        exponent = 1 / self._nu - 1
        norm = torch.pow(torch.linalg.vector_norm(gradient), exponent)
        self._point = self._point - self._learning_rate * norm * gradient
    
    def get_point(self):
        return self._point


class AdaptiveGradientDescent(BaseOptimizer):
    def __init__(self, model, point, lr, stable=False, delta=1e-8):
        self._model = model
        self._point = point
        self._gradient = self._model.gradient(self._point)
        self._learning_rate = lr
        self._stable = stable
        self._delta = delta
    
    def step(self):
        prev_func_value, _ = self._model.last_loss_and_accuracy()
        attempt_number = 1
        while True:
            # print("Attempt: {}, Learning Rate: {}".format(attempt_number, self._learning_rate))
            new_point = self._point - self._learning_rate * self._gradient
            new_gradient = self._model.gradient(new_point)
            func_value, _ = self._model.last_loss_and_accuracy()
            if not self._stable:
                if func_value <= (prev_func_value + 
                                torch.dot(self._gradient, new_point - self._point) + 
                                (1 / (2. * self._learning_rate)) * torch.linalg.vector_norm(new_point - self._point) ** 2):
                    break
            else:
                if (func_value <= prev_func_value - 
                    (self._learning_rate / 2.) * torch.linalg.vector_norm(self._gradient) ** 2 + self._delta):
                    break
            self._learning_rate /= 2
            attempt_number += 1
        self._point = new_point
        self._gradient = new_gradient
        self._learning_rate *= 2
    
    def get_point(self):
        return self._point


class AdaptiveHolderGradientDescent(BaseOptimizer):
    def __init__(self, model, point, lr, nu=1.0):
        self._model = model
        self._point = point
        self._learning_rate = lr
        self._nu = nu
    
    def step(self):
        prev_gradient = self._model.gradient(self._point)
        prev_func_value, _ = self._model.last_loss_and_accuracy()
        while True:
            exponent = 1 / self._nu - 1
            norm = torch.pow(torch.linalg.vector_norm(prev_gradient), exponent)
            new_point = self._point - self._learning_rate * norm * prev_gradient
            self._model.gradient(new_point)
            func_value, _ = self._model.last_loss_and_accuracy()
            if func_value <= (prev_func_value + 
                              torch.dot(prev_gradient, new_point - self._point) + 
                              (1 / ((1 + self._nu) * (self._learning_rate ** self._nu))) * torch.linalg.vector_norm(new_point - self._point) ** (1 + self._nu)):
                break
            self._learning_rate /= 2
        self._point = new_point
        self._learning_rate *= 2
    
    def get_point(self):
        return self._point


class SGD(BaseOptimizer):
    def __init__(self, model, point, lr, momentum=None):
        self._model = model
        self._point = point
        self._learning_rate = lr
    
    def step(self):
        stochastic_gradient = self._model.stochastic_gradient(self._point)
        self._point = self._point - self._learning_rate * stochastic_gradient
    
    def get_point(self):
        return self._point


class MomentumVarianceReduction(BaseOptimizer):
    def __init__(self, model, point, lr, momentum):
        self._model = model
        self._point = point
        self._gradient_estimator = 0
        self._learning_rate = lr
        self._momentum = momentum
        
    def init_gradient_estimator(self):
        number_of_batches = int(1 / self._momentum)
        assert number_of_batches > 0
        self._gradient_estimator = 0
        for _ in range(number_of_batches):
            gradient = self._model.stochastic_gradient(self._point)
            self._gradient_estimator = self._gradient_estimator + gradient
        self._gradient_estimator = self._gradient_estimator / np.float32(number_of_batches)
    
    def step(self):
        previous_point = self._point
        self._point = previous_point - self._learning_rate * self._gradient_estimator
        previous_gradient, current_gradient = self._model.stochastic_gradient_at_points([previous_point, self._point])
        self._gradient_estimator = current_gradient + (1 - self._momentum) * (self._gradient_estimator - previous_gradient)

    def get_point(self):
        return self._point
