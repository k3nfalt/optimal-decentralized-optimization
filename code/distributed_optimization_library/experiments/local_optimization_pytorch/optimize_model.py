import os
from pickletools import optimize
from statistics import variance
import yaml
import multiprocessing
import argparse
import time
import numpy as np
import random
import json

import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

from distributed_optimization_library.models.resnet import ResNet18
from distributed_optimization_library.models.resnet_without_bn import prepare_resnet_without_bn
from distributed_optimization_library.models.small_models import prepare_two_layer_nn
from distributed_optimization_library.models.small_models_eos import prepare_two_layer_nn_eos
from distributed_optimization_library.models.small_models_paper import prepare_two_layer_nn_paper
from distributed_optimization_library.function import parameters_to_tensor, tensor_to_parameters, BaseStochasticFunction

from distributed_optimization_library.experiments.local_optimization_pytorch.optimizers import GradientDescent, SignGradientDescent, HolderGradientDescent, AdaptiveGradientDescent, \
    AdaptiveHolderGradientDescent, SGD, MomentumVarianceReduction
from distributed_optimization_library.experiments.local_optimization_pytorch.logistic_regression import DebugMaskOneLogisticRegressionGradientDescent

_LARGE_NUMBER = 10**12


def save_dump(path, dump):
    path_tmp = path + "_tmp_"
    with open(path_tmp, 'w') as fd:
        json.dump(dump, fd)
    os.replace(path_tmp, path)


def save_model(path, model, batch_index):
    path_tmp = path + "_tmp_model_"
    torch.save(model.state_dict(), path_tmp)
    os.replace(path_tmp, path + "_model_{}".format(batch_index))


class StochasticModel(BaseStochasticFunction):
    def __init__(self, model, loss_fn, dataset, batch_size=1, num_workers=1, use_cuda=False,
                 optimize_memory=True):
        self._model = model
        self._use_cuda = use_cuda
        if use_cuda:
            self._model = self._model.cuda()
        self._loss_fn = loss_fn
        self._dataset = dataset
        self._batch_size = batch_size
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, 
            prefetch_factor=4,
            sampler=torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=_LARGE_NUMBER))
        self._dataloader_iter = iter(dataloader)
        self._num_workers = num_workers
        self._features_all = None
        self._optimize_memory = optimize_memory
        
        self._model.eval()
        features, _ = next(self._dataloader_iter)
        if self._use_cuda:
            features = features.cuda()
        self._model(features)
        self._model.train()
        print(self._model)

    def stochastic_gradient_at_points(self, points):
        current_seed = torch.seed()
        stochatic_gradients = []
        features, labels = next(self._dataloader_iter)
        if self._use_cuda:
            features, labels = features.cuda(), labels.cuda()
        for point in points:
            torch.manual_seed(current_seed)
            tensor_to_parameters(self._model.parameters(), point)
            loss, pred, stochatic_gradient = self._gradient(features, labels)
            stochatic_gradients.append(stochatic_gradient)
        self._last_loss = loss
        self._last_accuracy = self._accuracy(pred, labels)
        return stochatic_gradients
    
    def gradient(self, point, batch_size=None):
        tensor_to_parameters(self._model.parameters(), point)
        if not self._optimize_memory:
            if self._features_all is None:
                self._dataloader = torch.utils.data.DataLoader(self._dataset, 
                                                               batch_size=len(self._dataset), 
                                                               num_workers=self._num_workers,
                                                               prefetch_factor=4)
                for features, labels in self._dataloader:
                    pass
                self._features_all = features
                self._labels_all = labels
                if self._use_cuda:
                    self._features_all = self._features_all.cuda()
                    self._labels_all = self._labels_all.cuda()
            loss, pred, gradient = self._gradient(self._features_all, self._labels_all)
            self._last_loss = loss
            self._last_accuracy = self._accuracy(pred, self._labels_all)
        else:
            batch_size = batch_size if batch_size is not None else self._batch_size
            dataloader = torch.utils.data.DataLoader(self._dataset, 
                                                     batch_size=batch_size, 
                                                     num_workers=self._num_workers,
                                                     prefetch_factor=4) 
            total_number = 0
            sum_gradients = 0
            total_loss = 0
            total_labels = []
            total_pred = []
            for features, labels in dataloader:
                if self._use_cuda:
                    features, labels = features.cuda(), labels.cuda()
                loss, pred, gradient = self._gradient(features, labels)
                total_loss += loss * len(labels)
                total_labels.append(labels)
                total_pred.append(pred)
                gradient = gradient * len(labels)
                sum_gradients = (sum_gradients + gradient).detach()
                total_number += len(labels)
            gradient = sum_gradients / total_number
            self._last_loss = total_loss / total_number
            total_labels = torch.cat(total_labels)
            total_pred = torch.cat(total_pred)
            self._last_accuracy = self._accuracy(total_pred, total_labels)
        return gradient
    
    def hessian(self, point):
        assert not self._optimize_memory
        assert self._features_all is not None
        tensor_to_parameters(self._model.parameters(), point)
        parameters = tuple(self._model.parameters())
        parameters_shape = [parameter.shape for parameter in parameters]
        parameters = tuple(torch.flatten(parameter) for parameter in parameters)
        def hack_model(*params):
            params = [torch.reshape(param, shape) for param, shape in zip(params, parameters_shape)]
            names = list(n for n, _ in self._model.named_parameters())
            pred = torch.nn.utils.stateless.functional_call(self._model, 
                                                            {n: p for n, p in zip(names, params)}, 
                                                            self._features_all)
            loss = self._loss_fn(pred, self._labels_all)
            return loss
        hessians = torch.autograd.functional.hessian(hack_model, parameters)
        hessians = [torch.concat(h, axis=1) for h in hessians]
        hessian = torch.concat(hessians, axis=0)
        return hessian

    def parameters(self):
        return self._model.parameters()
    
    def current_point(self):
        return parameters_to_tensor(self._model.parameters())
    
    def last_loss_and_accuracy(self):
        return self._last_loss, self._last_accuracy
    
    def _gradient(self, features, labels):
        pred = self._model(features)
        loss = self._loss_fn(pred, labels)
        self._model.zero_grad()
        loss.backward()
        gradient = parameters_to_tensor(self._model.parameters(), grad=True)
        return loss, pred, gradient

    def _accuracy(self, pred, labels):
        if pred.shape[-1] > 1:
            _, predicted = pred.max(1)
        else:
            predicted = torch.squeeze(pred, dim=-1) >= 0.0
        accuracy = predicted.eq(labels).sum().item() / labels.size(0)
        return accuracy


def mean_and_sigma_2(model, batch_size_sigma, config):
    point = model.current_point()
    batch_size = config.get('batch_size_save_every', config['batch_size'])
    mean_gradient = model.gradient(point, batch_size=batch_size).cpu().numpy()
    norm_mean_gradient = np.linalg.norm(mean_gradient)
    estimated_variance = 0
    for _ in range(batch_size_sigma):
        gradient = model.stochastic_gradient(point).cpu().numpy()
        estimated_variance += np.linalg.norm(gradient - mean_gradient) ** 2
    estimated_variance /= batch_size_sigma
    return norm_mean_gradient, estimated_variance


class TransformedDataset(Dataset):
    def __init__(self, original_dataset, label_transform):
        self.original_dataset = original_dataset
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        data, label = self.original_dataset[idx]
        transformed_label = self.label_transform(label)
        return data, transformed_label


def prepare_dataset(path_to_dataset, number_of_first_examples=None, binary_classification=False,
                    filter_classes=None, resize=None, dataset='cifar10'):
    if dataset == 'cifar10':
        transform_train = [transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    elif dataset == 'mnist':
        transform_train = [transforms.ToTensor(),
                           transforms.Normalize((0.5), (1.0))]
    elif dataset == 'flowers102':
        raise RuntimeError()
    elif dataset == 'fashion_mnist':
        transform_train = [transforms.ToTensor(),
                           transforms.Normalize((0.5), (1.0))]
    if resize is not None:
        transform_train.append(transforms.Resize(size=resize))
    transform_train = transforms.Compose(transform_train)
    target_transform = None
    if binary_classification:
        assert filter_classes is None
        target_transform = lambda y: int(y >= 5)
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=path_to_dataset, train=True, download=True,
                                                transform=transform_train,
                                                target_transform=target_transform)
    elif dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=path_to_dataset, train=True, download=True,
                                                transform=transform_train,
                                                target_transform=target_transform)
    elif dataset == 'flowers102':
        trainset = torchvision.datasets.Flowers102(root=path_to_dataset, split='train', download=True,
                                                   transform=transform_train,
                                                   target_transform=target_transform)
    elif dataset == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(root=path_to_dataset, train=True, download=True,
                                                   transform=transform_train,
                                                   target_transform=target_transform)
    if filter_classes is not None:
        labels = np.array(trainset.targets, dtype=np.int32)
        mask = np.zeros(len(labels), dtype=np.bool)
        for filter_class in filter_classes:
            mask[labels == filter_class] = True
        indices = np.where(mask)[0]
        trainset = torch.utils.data.Subset(trainset, indices)
        _map_labels = {label: index for index, label in enumerate(filter_classes)}
        def transform_label(label):
            return _map_labels[label]
        trainset = TransformedDataset(trainset, transform_label)
    if number_of_first_examples is not None:
        trainset = torch.utils.data.Subset(trainset, range(number_of_first_examples))
        
    labels = [label for _, label in trainset]
    for sample, _ in trainset:
        pass
    num_of_features = np.prod(sample.shape)
    number_of_channels = sample.shape[0]
    stats_labels = np.unique(labels, return_counts=True)
    print(f"Stats Labels: {stats_labels}")
    print(f"Number of shape/min/max: {sample.shape,sample.min(),sample.max()}")
    number_of_classes = len(np.unique(labels))
    print(f"Number of classes: {number_of_classes}")
    return trainset, number_of_classes, num_of_features, number_of_channels


def run_experiments(path_to_dataset, dumps_path, config, basename):
    if config.get('use_double', False):
        torch.set_default_tensor_type(torch.DoubleTensor)
    dump_path = os.path.join(dumps_path, basename)
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if config.get('number_of_samples', None) is not None:
        number_of_first_examples = config['number_of_samples']
    else:
        number_of_first_examples = 5000 if config.get('first_5000_examples', False) else None
    trainset, number_of_classes, num_of_features, number_of_channels = prepare_dataset(
        path_to_dataset, number_of_first_examples=number_of_first_examples,
        binary_classification=config.get('binary_classification', False),
        filter_classes=config.get('filter_classes', None),
        dataset=config.get('dataset', 'cifar10'))
    
    model_name = config.get('model', 'resnet')
    optimize_memory = True
    loss_function_name = config.get('loss_function', 'cross_entropy')
    if loss_function_name == 'bce_logits':
        assert number_of_classes == 2
        number_of_outputs = 1
    else:
        number_of_outputs = number_of_classes
    if model_name == 'resnet':
        model = ResNet18(activation_name=config['resnet_params'].get('activation', 'relu'))
        optimize_memory = config.get('optimize_memory', True)
    elif model_name == 'resnet_without_bn':
        model = prepare_resnet_without_bn(**config['resnet_params'],
                                          num_classes=number_of_outputs,
                                          number_of_channels=number_of_channels)
        optimize_memory = config.get('optimize_memory', True)
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v3_small()
        optimize_memory = config.get('optimize_memory', True)
    elif model_name == 'two_layer_nn':
        model = prepare_two_layer_nn(**config['two_layer_nn_params'])
        optimize_memory = False
    elif model_name == 'two_layer_nn_eos':
        model = prepare_two_layer_nn_eos(**config['two_layer_nn_params_eos'],
                                         number_of_outputs=number_of_outputs,
                                         number_of_features=num_of_features,
                                         number_of_channels=number_of_channels)
        optimize_memory = False
    elif model_name == 'two_layer_nn_paper':
        model = prepare_two_layer_nn_paper(**config['two_layer_nn_params_paper'],
                                           number_of_outputs=number_of_outputs,
                                           number_of_features=num_of_features,
                                           number_of_channels=number_of_channels)
        optimize_memory = config.get('optimize_memory', True)
    cudnn.benchmark = True
    if loss_function_name == 'cross_entropy':
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss_function_name == 'bce_logits':
        assert number_of_outputs == 1
        loss_nn = torch.nn.BCEWithLogitsLoss()
        def _bce_logits_loss(scores, labels):
            assert scores.shape[-1] == 1
            scores = torch.squeeze(scores, dim=-1)
            labels = labels.float()
            return loss_nn(scores, labels)
        loss_fn = _bce_logits_loss
    elif loss_function_name in ['mse', 'smooth_l1']:
        if loss_function_name == 'mse':
            loss_nn = torch.nn.MSELoss()
        else:
            loss_nn = torch.nn.SmoothL1Loss()
        loss_fn = lambda scores, labels: loss_nn(scores, torch.nn.functional.one_hot(labels).float())
        
    model_wrapper = StochasticModel(model, loss_fn, trainset, 
                                    batch_size=config['batch_size'], 
                                    num_workers=config.get('num_workers', 1),
                                    use_cuda=config.get('use_cuda', False),
                                    optimize_memory=optimize_memory)
    point = model_wrapper.current_point()
    if config['optimizer'] == 'sgd':
        optimizer_wrapper = SGD(model_wrapper, point, lr=config['learning_rate'],
                                momentum=config['momentum'])
    elif config['optimizer'] == 'mvr':
        optimizer_wrapper = MomentumVarianceReduction(model_wrapper, point, lr=config['learning_rate'],
                                                      momentum=config['momentum'])
        if config.get('init_gradient_estimator', False):
            optimizer_wrapper.init_gradient_estimator()
    elif config['optimizer'] == 'gd':
        optimizer_wrapper = GradientDescent(model_wrapper, point, lr=config['learning_rate'])
    elif config['optimizer'] == 'debug_gd':
        optimizer_wrapper = DebugMaskOneLogisticRegressionGradientDescent(model_wrapper, point, lr=config['learning_rate'])
    elif config['optimizer'] == 'sign_gd':
        optimizer_wrapper = SignGradientDescent(model_wrapper, point, lr=config['learning_rate'])
    elif config['optimizer'] == 'adaptive_gd':
        optimizer_wrapper = AdaptiveGradientDescent(model_wrapper, point, lr=10**9)
    elif config['optimizer'] == 'adaptive_stable_gd':
        optimizer_wrapper = AdaptiveGradientDescent(model_wrapper, point, lr=10**9, stable=True, 
                                                    delta=config.get('adaptive_stable_gd_delta', 1e-8))
    elif config['optimizer'] == 'holder_gd':
        optimizer_wrapper = HolderGradientDescent(model_wrapper, point, lr=config['learning_rate'],
                                                  nu=config['holder_constant'])
    elif config['optimizer'] == 'adaptive_holder_gd':
        optimizer_wrapper = AdaptiveHolderGradientDescent(model_wrapper, point, lr=10**9,
                                                          nu=config['holder_constant'])
    stat = {'batch_loss': [], 'batch_accuracy': [], 'batch_norm_of_gradient': [],
            'estimate_sigma_2': [], 'estimate_norm_of_gradient': [],
            'norm_of_gradient': [], 'sigma_2': [],
            'batch_diff_gradient': [], 'batch_diff_points': [],
            'batch_dot_gradient': [], 'learning_rates': [], 'extra_stat': []}
    dump = {'config': config, 'stat': stat}
    batch_index = 0
    while True:
        start_time = time.time()
        if batch_index > 0:
            prev_gradient = parameters_to_tensor(model.parameters(), grad=True).cpu().numpy()
        point = optimizer_wrapper.get_point().cpu().numpy()
        optimizer_wrapper.step()
        iter_time = time.time() - start_time
        gradient = parameters_to_tensor(model.parameters(), grad=True).cpu().numpy()
        norm_of_gradient = np.linalg.norm(gradient)
        if batch_index > 0:
            diff_gradient = np.linalg.norm(gradient - prev_gradient)
            dot_gradient = np.dot(gradient, prev_gradient)
            diff_points = np.linalg.norm(point - prev_point)
        prev_point = point
        loss, accuracy = model_wrapper.last_loss_and_accuracy()
        stat['batch_loss'].append(loss.item())
        stat['batch_norm_of_gradient'].append(norm_of_gradient.item())
        stat['batch_accuracy'].append(accuracy)
        stat['learning_rates'].append(optimizer_wrapper._learning_rate)
        stat['extra_stat'].append(optimizer_wrapper.get_stat())
        if batch_index > 0:
            stat['batch_diff_gradient'].append(diff_gradient.item())
            stat['batch_dot_gradient'].append(dot_gradient.item())
            stat['batch_diff_points'].append(diff_points.item())
        print("iter: {iter} / {number_of_batches} loss: {loss}, norm_of_gradient**2 :{norm_of_gradient} acc: {accuracy}, time: {iter_time}".format(
              loss=loss, iter_time=iter_time, accuracy=accuracy, norm_of_gradient=norm_of_gradient**2,
              iter=batch_index, number_of_batches=config['number_of_batches']))
        if batch_index >= config['number_of_batches']:
            save_model(dump_path, model, batch_index)
            save_dump(dump_path, dump)
            break
        if config.get('save_model_every', None) is not None and batch_index % config['save_model_every'] == 0:
            save_model(dump_path, model, batch_index)
        if batch_index % config.get('save_every', 100) == 0:
            save_dump(dump_path, dump)
            if config.get('mean_and_sigma', False):
                norm_mean_gradient, variance = mean_and_sigma_2(model_wrapper, batch_size_sigma=100, config=config)
                stat['norm_of_gradient'].append((batch_index, norm_mean_gradient.item()))
                stat['sigma_2'].append((batch_index, variance.item()))
        batch_index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--path_to_dataset')

    args = parser.parse_args()
    basename = os.path.basename(args.config).split('.')[0]
    configs = yaml.safe_load(open(args.config))
    assert len(configs) == 1
    config = configs[0]
    if config.get('parallel', False):
        torch.multiprocessing.set_sharing_strategy('file_system')
        if config.get('multiprocessing_spawn', False):
            multiprocessing.set_start_method("spawn")
    run_experiments(args.path_to_dataset, args.dumps_path, config, basename)

if __name__ == "__main__":
    main()
