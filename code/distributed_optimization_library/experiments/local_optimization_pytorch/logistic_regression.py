import torch 

from distributed_optimization_library.experiments.local_optimization_pytorch.optimizers import BaseOptimizer
from distributed_optimization_library.models.small_models_eos import circulant

def _prepare_value(v):
    return v.detach().cpu().numpy().tolist()


class DebugMaskOneLogisticRegressionGradientDescent(BaseOptimizer):
    def __init__(self, model, point, lr):
        self._model = model
        self._point = point
        self._learning_rate = lr
        self._learning_rate_mirror = torch.tensor(lr, requires_grad=True)
        self._mirror_a = None
        self._mirror_b = None
        self._recursion_history = 100
        self._current_history = 0
        self._init_b = None
        self._cum_h = 0
    
    def step(self):
        gradient = self._model.gradient(self._point)
        loss, _ = self._model.last_loss_and_accuracy()
        explicit_loss = self._explicit_loss()
        # assert torch.abs(loss - explicit_loss) < 1e-3
        self._point = self._point - self._learning_rate * gradient
        # self._mirror_update()
        # self._grad_wrt_lr()
    
    def get_point(self):
        return self._point
    
    def _explicit_loss(self):
        parameters = self._model._model.parameters()
        parameters = list(parameters)
        mask_features = self._model._model[1]._mask_features
        a = parameters[0][:mask_features]
        b = parameters[1]
        if self._init_b is None:
            self._init_b = b
        if self._mirror_a is None:
            self._mirror_a = a.clone().detach().requires_grad_(False)
            self._mirror_b = b.clone().detach().requires_grad_(False).view(-1)
        self._current_a = _prepare_value(a)
        norm_b = torch.norm(b)
        self._current_norm_b = _prepare_value(norm_b)
        h = self._calculate_h(a, b)
        p_matrix = torch.zeros_like(b.flatten())
        p_matrix[-1] = 1
        p_matrix = circulant(p_matrix)
        norm_h = torch.norm(h)
        self._current_norm_h = _prepare_value(norm_h)
        sqrt_dot_h_P_h = torch.sqrt(torch.dot(h, p_matrix.matmul(h)))
        self._current_sqrt_dot_h_P_h = _prepare_value(sqrt_dot_h_P_h)
        loss = self._calculate_loss(a, b)
        
        samples = self._get_samples()
        a_circulant = self._a_circulant(a, samples)
        first_vec = torch.dot(h, b.flatten()) * (h @ a_circulant.T)
        second_vec = torch.dot(h, p_matrix.T @ b.flatten()) * (h @ a_circulant.T @ p_matrix)
        final_vec = first_vec + second_vec
        # final_vec = first_vec
        # print(final_vec)
        print(a)
        print(final_vec @ a_circulant.T @ b.flatten())
        # print(a ** 2 * (torch.dot(h, b.flatten()))**2)
        # self._cum_h = self._cum_h + h.detach()
        # print(torch.dot(self._cum_h.flatten(), self._init_b.flatten()))
        return loss
    
    def _grad_wrt_lr(self):
        assert False
        mirror_loss = self._calculate_loss(self._mirror_a, self._mirror_b)
        if self._learning_rate_mirror.grad is not None:
            self._learning_rate_mirror.grad.zero_()
        if self._current_history == self._recursion_history:
            self._mirror_a = self._mirror_a.clone().detach().requires_grad_(False)
            self._mirror_b = self._mirror_b.clone().detach().requires_grad_(False).view(-1)
            self._current_history = 0
        mirror_loss.backward(retain_graph=True)
        self._current_history += 1
        print(self._current_history, self._learning_rate_mirror.grad)
    
    def _get_samples(self):
        labels = 2 * self._model._labels_all - 1
        return self._model._features_all.view(self._model._features_all.shape[0], -1) * labels.view(-1, 1)
    
    def _calculate_loss(self, a, b):
        samples = self._get_samples()
        projections = self._projection(a, b, samples)
        loss_samples = torch.nn.functional.softplus(-projections)
        loss = torch.mean(loss_samples)
        return loss
    
    def _calculate_h(self, a, b):
        samples = self._get_samples()
        projections = self._projection(a, b, samples)
        weights = -torch.sigmoid(-projections)
        h = torch.mean(samples * weights.view(-1, 1), dim=0)
        return h
    
    def _projection(self, a, b, samples):
        a_circulant = self._a_circulant(a, samples)
        interm_output = samples.matmul(a_circulant.t())
        return interm_output.matmul(b.t())
    
    def _a_circulant(self, a, samples):
        num_feat = samples.shape[1]
        a_expand = torch.nn.functional.pad(a, (0, num_feat - len(a)), "constant", 0)
        a_circulant = circulant(a_expand)
        return a_circulant

    def _mirror_update(self):
        a = self._mirror_a
        b = self._mirror_b
        h = self._calculate_h(a, b)
        self._mirror_a = self._mirror_a - self._learning_rate_mirror * torch.dot(h, self._mirror_b)
        self._mirror_b = self._mirror_b - self._learning_rate_mirror * h * self._mirror_a

    def get_stat(self):
        return {'a': self._current_a,
                'norm_b': self._current_norm_b,
                'norm_h': self._current_norm_h,
                'sqrt_dot_h_P_h': self._current_sqrt_dot_h_P_h}
