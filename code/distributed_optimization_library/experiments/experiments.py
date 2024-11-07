import os
import argparse
import json
from posixpath import basename
import statistics
import uuid
import random
from scipy.sparse import data
import yaml
import math
import time
import multiprocessing
from io import BytesIO

import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch

from distributed_optimization_library.function import NonConvexLossFunction, LogisticRegressionFunction, Resnet18Function, \
    NonConvexLossMultiClassFunction, LogisticRegressionOptimizationProblemMeta
from distributed_optimization_library.function import generate_random_vector, QuadraticFunction, TridiagonalQuadraticFunction, \
    OptimizationProblemMeta, TridiagonalQuadraticOptimizationProblemMeta, NeuralNetworkFunction, AutoEncoderNeuralNetworkFunction, \
    StochasticLogisticRegressionFunction, StochasticTridiagonalQuadraticFunction, StochasticMatrixTridiagonalQuadraticFunction, \
    MeanTridiagonalQuadraticFunction, SamplingType
from distributed_optimization_library.algorithm import get_algorithm as get_comm_algorithm
from distributed_optimization_library.asynchronous.algorithm import get_algorithm as get_async_algorithm
from distributed_optimization_library.asynchronous.algorithm_with_communication import get_algorithm as get_async_algorithm_with_communication
from distributed_optimization_library.asynchronous.algorithm_with_graphs import get_algorithm as get_async_algorithm_with_graphs
from distributed_optimization_library.dataset import LibSVMDataset, MNISTDataset, Dataset
from distributed_optimization_library.transport import find_total, find_total_node
from distributed_optimization_library.signature import Signature


def serialize_point(point):
    if torch.is_tensor(point):
        point = point.cpu().numpy()
    memfile = BytesIO()
    np.save(memfile, point)
    memfile.seek(0)
    return memfile.read().decode('latin-1')


class OptimizerClassification(object):
    def _init_classification_model(self):
        if self._params['config']['task'] == 'libsvm':
            dataset_train = self._params['function_stats']['recovered_datset']
            del self._params['function_stats']['recovered_datset']
            if self._params['config']['dataset_name'] == 'mnist':
                dataset_test = MNISTDataset(
                    os.path.join(self._params['path_to_dataset'], "digit-recognizer"), train=False)
            elif self._params['config']['dataset_name'] == 'cifar10':
                dataset_test = LibSVMDataset.from_file(self._params['path_to_dataset'], 
                                                       self._params['config']['dataset_name'],
                                                       return_sparse=self._params['config'].get('sparse_dataset', False),
                                                       train=False)
            else:
                dataset_test = None
            log_regresion_test = None
            if self._params['config']['function'] in ['logistic_regression', 'nonconvex_multiclass', 'stochastic_logistic_regression']:
                params = {'number_of_classes': dataset_train.number_of_classes()}
                log_regresion_train = LogisticRegressionFunction(*dataset_train.get_data_and_labels(), **params)
                if dataset_test is not None:
                    log_regresion_test = LogisticRegressionFunction(*dataset_test.get_data_and_labels(), **params)
                self._quality_check = True
            elif self._params['config']['function'] in ['nonconvex']:
                log_regresion_train = NonConvexLossFunction(*dataset_train.get_data_and_labels(), seed=42)
                log_regresion_test = None
                self._quality_check = True
            elif 'two_layer_neural' in self._params['config']['function']:
                params = {'number_of_classes': dataset_train.number_of_classes()}
                params['neural_network_name'] = self._params['config']['function']
                log_regresion_train = NeuralNetworkFunction(*dataset_train.get_data_and_labels(), **params)
                if dataset_test is not None:
                    log_regresion_test = NeuralNetworkFunction(*dataset_test.get_data_and_labels(), **params)
                self._quality_check = True
            elif 'auto_encoder' in self._params['config']['function']:
                params = {'neural_network_name': self._params['config']['function'],
                            'encode_dim': self._params['config'].get('encode_dim', 16)}
                log_regresion_train = AutoEncoderNeuralNetworkFunction(*dataset_train.get_data_and_labels(), **params)
                log_regresion_test = AutoEncoderNeuralNetworkFunction(*dataset_test.get_data_and_labels(), **params)
                self._quality_check = True
        if self._params['config']['task'] == 'cifar10':
            log_regresion_train = None
            log_regresion_test = self._params['function_stats']['function_test']
            self._quality_check = True
        if self._quality_check:
            self._log_regresion_train = log_regresion_train
            self._log_regresion_test = log_regresion_test
            self._stat['accuracy_train'] = []
            self._stat['accuracy_test'] = []


class OptimizerAsyncStat(OptimizerClassification):
    def __init__(self, optimizer, params):
        self._optimizer = optimizer
        self._stat = {'times': []}
        self._params = params
        self._index = 0
        self._quality_check = False
        self._accuracy_train = None
        self._accuracy_test = None
        if self._params['config'].get('calculate_function', False):
            self._stat['function_values'] = []
        if self._params['config'].get('calculate_accuracy', False):
            self._init_classification_model()
        self._stat['number_of_iterations'] = 0
        self._stop_time = self._params['config'].get('stop_time', None)

        
    def step(self):
        self._optimizer.step()
        if self._index % self._params['config'].get('quality_check_rate', 1) == 0:
            if self._params['config'].get('calculate_function', False):
                function_value = float(self._optimizer.calculate_function())
                self._stat['function_values'].append(function_value)
                print("Function value: {}".format(function_value))
            if self._quality_check:
                if self._log_regresion_train is not None:
                    self._accuracy_train = self._log_regresion_train._check_accuracy(self._optimizer.get_point())
                    self._stat['accuracy_train'].append(self._accuracy_train)
                if self._log_regresion_test is not None:
                    self._accuracy_test = self._log_regresion_test._check_accuracy(self._optimizer.get_point())
                    self._stat['accuracy_test'].append(self._accuracy_test)
                print("Acc Train: {}, Acc Test: {}".format(self._accuracy_train, self._accuracy_test))
            self._stat['times'].append(self._optimizer.get_time())
            if self._params['config'].get('calculate_function', False):
                if math.isnan(function_value):
                    return False
        self._index += 1
        self._stat['number_of_iterations'] = self._index
        if (self._stop_time is not None and len(self._stat['times']) > 0 and self._stop_time < self._stat['times'][-1]):
            return False
        return True
    
    def dump(self, path, name):
        dm = {'stat': self._stat, 'params': self._params}
        dm['point'] = serialize_point(self._optimizer.get_point())
        with open(os.path.join(path, name), 'w') as fd:
            json.dump(dm, fd)


class OptimizerStat(OptimizerClassification):
    def __init__(self, optimizer, params):
        self._optimizer = optimizer
        self._stat = {'bites_send_to_nodes': [],
                      'bites_send_from_nodes': [],
                      'max_bites_send_from_nodes': [],
                      'similarity_characteristics': []}
        self._params = params
        self._index = 0
        self._quality_check = False
        self._start_time = time.time()
        if self._params['config'].get('calculate_accuracy', False):
            self._init_classification_model()
        self._accuracy_train = None
        self._accuracy_test = None
        if self._params['config'].get('calculate_function', False):
            self._stat['function_values'] = []
        if self._params['config'].get('calculate_norm_of_gradients', False):
            self._stat['norm_of_gradients'] = []
        if self._params['config'].get('calculate_gradient_estimator_error', False):
            self._stat['gradient_estimator_error'] = []
        if self._params['config'].get('calculate_smoothness_variance', False):
            self._stat['smoothness_variance'] = []
        if self._params['config'].get('statistics', False):
            self._stat['statistics'] = []
        self._stat['number_of_iterations'] = 0

        
    def step(self):
        self._optimizer.step()
        with self._optimizer.ignore_statistics():
            if self._index % self._params['config'].get('quality_check_rate', 1) == 0:
                if self._params['config'].get('statistics', False):
                    statistics = self._optimizer.statistics()
                    print(statistics)
                    self._stat['statistics'].append(statistics)
                if self._params['config'].get('calculate_function', False):
                    function_value = float(self._optimizer.calculate_function())
                    self._stat['function_values'].append(function_value)
                    print("Function value: {}".format(function_value))
                if self._quality_check:
                    if self._log_regresion_train is not None:
                        self._accuracy_train = self._log_regresion_train._check_accuracy(self._optimizer.get_point())
                        self._stat['accuracy_train'].append(self._accuracy_train)
                    if self._log_regresion_test is not None:
                        self._accuracy_test = self._log_regresion_test._check_accuracy(self._optimizer.get_point())
                        self._stat['accuracy_test'].append(self._accuracy_test)
                print("Accuracy. train: {}; test: {}".format(self._accuracy_train, self._accuracy_test))
                gradient = None
                if self._params['config'].get('calculate_norm_of_gradients', False):
                    gradient = self._optimizer.calculate_gradient()
                    norm_of_gradient = np.linalg.norm(gradient)
                    print("Norm of gradient: {}".format(norm_of_gradient))
                    self._stat['norm_of_gradients'].append(float(norm_of_gradient))
                if self._params['config'].get('calculate_gradient_estimator_error', False):
                    if gradient is None:
                        gradient = self._optimizer.calculate_gradient()
                    gradient_estimator = self._optimizer._gradient_estimator
                    diff = np.linalg.norm(gradient - gradient_estimator)
                    print("Diff: {}, Grad norm: {}".format(diff, np.linalg.norm(gradient)))
                    self._stat['gradient_estimator_error'].append(float(diff))
                if self._params['config'].get('calculate_smoothness_variance', False):
                    mean_norm, norm_mean = self._optimizer.calculate_smoothness_variance()
                    print("Mean norm: {} Norm mean: {}".format(mean_norm, norm_mean))
                    self._stat['smoothness_variance'].append((float(mean_norm), float(norm_mean)))
                
                stat_to_nodes, stat_from_nodes = self._optimizer.get_stats()
                max_stat_from_nodes = self._optimizer.get_max_stats()
                self._stat['bites_send_to_nodes'].append(find_total(stat_to_nodes))
                self._stat['bites_send_from_nodes'].append(find_total(stat_from_nodes))
                self._stat['max_bites_send_from_nodes'].append(find_total_node(max_stat_from_nodes))
                if self._params['config'].get('calculate_function', False):
                    if math.isnan(function_value):
                        return False
                    if self._params['config'].get('early_stop_function', None) is not None:
                        if function_value <= self._params['config']['early_stop_function']:
                            return False
            if self._index % 1000 == 0:
                stat_to_nodes, stat_from_nodes = self._optimizer.get_stats()
                max_stat_from_nodes = self._optimizer.get_max_stats()
                print(stat_from_nodes)
                print(max_stat_from_nodes)
            self._index += 1
            self._stat['number_of_iterations'] = self._index
        return True
    
    def dump(self, path, name):
        dm = {'stat': self._stat, 'params': self._params,
              'start_time': self._start_time,
              'end_time': time.time()}
        if self._params['config']['task'] != 'cifar10':
            with self._optimizer.ignore_statistics():
                dm['point'] = serialize_point(self._optimizer.get_point())
        else:
            if 'function_test' in self._params['function_stats']:
                del self._params['function_stats']['function_test']
        with open(os.path.join(path, name), 'w') as fd:
            json.dump(dm, fd)


def mean(vectors):
    return sum(vectors) / float(len(vectors))


def prepare_libsvm(path_to_dataset, config, generator):
    meta = OptimizationProblemMeta()
    if config['dataset_name'] == 'mnist':
        dataset = MNISTDataset(os.path.join(path_to_dataset, "digit-recognizer"))
    else:
        dataset = LibSVMDataset.from_file(path_to_dataset, config['dataset_name'],
                                          return_sparse=config.get('sparse_dataset', False))
    params = config.get('function_parameters', {})
    if config['shuffle']:
        dataset.shuffle(generator)
    if config.get('subsample_classes', None) is not None:
        dataset = dataset.subsample_classes(number_of_classes=config['subsample_classes'],
                                            seed=generator)
    if config['function'] == 'logistic_regression':
        func_cls = LogisticRegressionFunction
        params['sampling'] = config.get('sampling_name', SamplingType.UNIFORM_WITH_REPLACEMENT)
        params['number_of_classes'] = dataset.number_of_classes()
        params['reg_paramterer'] = config.get('reg_paramterer', 0.0)
        params['batch_size'] = config.get('batch_size', None)
    elif config['function'] == 'stochastic_logistic_regression':
        func_cls = StochasticLogisticRegressionFunction
        params['number_of_classes'] = dataset.number_of_classes()
        params['batch_size'] = config['batch_size']
    elif config['function'] == 'nonconvex':
        func_cls = NonConvexLossFunction
        params['seed'] = generator
        params['sampling'] = config.get('sampling_name', SamplingType.UNIFORM_WITH_REPLACEMENT)
        params['batch_size'] = config.get('batch_size', None)
    elif config['function'] == 'nonconvex_multiclass':
        func_cls = NonConvexLossMultiClassFunction
        params['number_of_classes'] = dataset.number_of_classes()
    elif 'two_layer_neural' in config['function']:
        func_cls = NeuralNetworkFunction
        params['reg_paramterer'] = config.get('reg_paramterer', 0.0)
        params['neural_network_name'] = config['function']
        params['number_of_classes'] = dataset.number_of_classes()
    elif 'auto_encoder' in config['function']:
        func_cls = AutoEncoderNeuralNetworkFunction
        params['reg_paramterer'] = config.get('reg_paramterer', 0.0)
        params['neural_network_name'] = config['function']
        params['encode_dim'] = config.get('encode_dim', 16)
        params['point_initializer'] = config.get('point_initializer', None)
        params['batch_size'] = config.get('batch_size', None)
    else:
        raise RuntimeError()
    function_stats = {}
    if config.get('equalize_to_same_number_samples_per_class', None) is not None:
        print("Equalize to same number samples per class")
        number_samples = config['equalize_to_same_number_samples_per_class']
        dataset = dataset.equalize_to_same_number_samples_per_class(number_samples)
        dataset.shuffle(generator)
    if not config.get('homogeneous', False):
        if config.get('split_with_controling_homogeneity', None) is not None:
            print("Split with controling homogeneity")
            split_with_controling_homogeneity = config['split_with_controling_homogeneity']
            assert split_with_controling_homogeneity > 0
            dataset_splits = dataset.split_with_controling_homogeneity(
                config['num_nodes'], prob_taking_from_hold_out=split_with_controling_homogeneity,
                seed=generator)
        elif config.get('split_with_all_dataset', None) is not None:
            print("Split with all dataset")
            split_with_all_dataset = config['split_with_all_dataset']
            split_with_all_dataset_max_number = config['split_with_all_dataset_max_number']
            assert split_with_all_dataset > 0
            dataset_splits = dataset.split_with_all_dataset(
                config['num_nodes'], prob_taking_all_dataset=split_with_all_dataset,
                seed=generator,
                max_number=split_with_all_dataset_max_number)
        elif config.get('split_into_groups_by_labels', False):
            print("Split into groups by labels")
            dataset_splits, nodes_indices_splits = dataset.split_into_groups_by_labels(config['num_nodes'])
            function_stats['nodes_indices_splits'] = nodes_indices_splits
        else:
            print("Split original")
            print("New labels: {}".format(np.unique(dataset.get_data_and_labels()[1], return_counts=True)))
            dataset_splits = dataset.split(config['num_nodes'], ignore_remainder=config.get('ignore_remainder', False))
        if config.get('calculate_accuracy', False):
            recovered_datset = Dataset.combine(dataset_splits)
            function_stats['recovered_datset'] = recovered_datset
        functions = []
        for dataset_ in dataset_splits:
            features, labels = dataset_.get_data_and_labels()
            params_copy = dict(params)
            params_copy['features'] = features
            params_copy['labels'] = labels
            functions.append(func_cls(**params_copy))
    else:
        functions = []
        features, labels = dataset.get_data_and_labels()
        if config.get('concat_ones_to_features', False):
            ones_features = np.ones((len(features), 1), dtype=np.float32)
            features = np.concatenate((features, ones_features), axis=1)
        for _ in range(config['num_nodes']):
            params_copy = dict(params)
            params_copy['features'] = features
            params_copy['labels'] = labels
            if config.get('no_copy_features', False):
                params_copy['no_copy_features'] = True
            functions.append(func_cls(**params_copy))
        if config.get('calculate_accuracy', False):
            function_stats['recovered_datset'] = dataset
    if config['function'] == 'logistic_regression':
        meta = LogisticRegressionOptimizationProblemMeta(functions)
    if config['function'] in ['logistic_regression', 'nonconvex', 'nonconvex_multiclass', 'stochastic_logistic_regression']:
        point = np.zeros((functions[0].dim(),), dtype=np.float32)
    elif 'two_layer_neural' in config['function'] or 'auto_encoder' in config['function']:
        point = functions[0].get_current_point()
    if config.get('scale_initial_point', None) is not None:
        point = config['scale_initial_point'] * point
    assert 'dim' not in config.keys() or len(point) == config['dim'], (len(point), config['dim'])
    if config.get('page_ab_gamma', False):
        assert len(functions) == 1
        def calculate_gamma(l_minus, prob, A, B, l_plus):
            assert A == B
            inv_gamma = l_minus + math.sqrt(((1 - prob) / prob) * (B * l_plus ** 2))
            return 1 / inv_gamma
        sampling = config.get('sampling_name', SamplingType.UNIFORM_WITH_REPLACEMENT)
        batch_size = config['algorithm_master_params']['batch_size']
        local_lipt = functions[0].liptschitz_local_gradient_constants()
        function_stats['number_of_functions'] = functions[0].number_of_functions()
        function_stats['local_lipt'] = local_lipt.tolist()
        if sampling == SamplingType.UNIFORM_WITH_REPLACEMENT:
            A = B = 1 / batch_size
            l_minus = np.mean(local_lipt)
            l_plus = np.sqrt(np.mean(local_lipt ** 2))
            prob = batch_size / (batch_size + functions[0].number_of_functions())
        elif config['sampling_name'] == SamplingType.IMPORTANCE:
            A = B = 1 / batch_size
            l_minus = np.mean(local_lipt)
            l_plus = np.mean(local_lipt)
            prob = batch_size / (batch_size + functions[0].number_of_functions())
        config['algorithm_master_params']['gamma'] = calculate_gamma(l_minus, prob, A, B, l_plus)
        
    return functions, point, function_stats, meta


def wrapper_resnet(node_index, path_to_dataset, augmentation, num_nodes, batch_size, seed, 
                   resnet_params={}):
    generator_torch = torch.Generator().manual_seed(42)
    if augmentation:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    trainset = torchvision.datasets.CIFAR10(root=path_to_dataset, train=True, download=True,
                                            transform=transform_train)
    number_of_exampels = len(trainset)
    number_of_exampels_per_node = number_of_exampels // num_nodes
    trainsets = torch.utils.data.random_split(trainset, 
                                              [number_of_exampels_per_node] * num_nodes,
                                              generator=generator_torch)
    trainset = trainsets[node_index]
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    return Resnet18Function(trainset, batch_size=batch_size // num_nodes, num_workers=2, seed=seed, **resnet_params)


def prepare_cifar10(path_to_dataset, config, generator):
    resnet_params = config.get('resnet_params', {})
    functions = [Signature(wrapper_resnet, 
                           node_index, path_to_dataset=path_to_dataset, 
                           augmentation=config.get('augmentation', False), 
                           num_nodes=config['num_nodes'], 
                           batch_size=config['batch_size'], 
                           seed=generator.integers(10e6),
                           resnet_params=resnet_params) 
                 for node_index in range(config['num_nodes'])]
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=path_to_dataset, train=False, download=True,
                                           transform=transform_test)
    function_only_for_point = Resnet18Function(testset, batch_size=1, num_workers=2, seed=generator.integers(10e6),
                                               **resnet_params)
    point = function_only_for_point.get_current_point()
    testset = torchvision.datasets.CIFAR10(root=path_to_dataset, train=False, download=True,
                                           transform=transform_test)
    function_test = Resnet18Function(testset, batch_size=100, num_workers=2, seed=generator,
                                     **resnet_params)
    return functions, point, {'function_test': function_test}, OptimizationProblemMeta()


def prepare_quadratic(path_to_dataset, config, generator):
    meta = OptimizationProblemMeta()
    if config.get('random', False):
        creator = QuadraticFunction.create_random
        if not config.get('homogeneous', False):
            functions = [creator(config['dim'], seed=generator) 
                         for _ in range(config['num_nodes'])]
        else:
            function = creator(config['dim'], seed=generator)
            functions = [function for _ in range(config['num_nodes'])]
    elif config.get('function') == 'square_norm':
        functions = [QuadraticFunction.create_norm_square(config['dim'])
                     for _ in range(config['num_nodes'])]
        if 'square_norm_lipt' in config:
            for i in range(config['num_nodes']):
                functions[i]._A = functions[i]._A * (1 + config['square_norm_lipt'][i])
    else:
        functions = TridiagonalQuadraticFunction.load_functions(
            os.path.join(config['dump_path'], "functions"))
        if config.get('type') == 'stochastic':
            noise = config['noise']
            type_noise = config.get('type_noise', 'add')
            functions = [StochasticTridiagonalQuadraticFunction.from_tridiagonal_quadratic
                         (f, generator, noise, type_noise) for f in functions]
        elif config.get('type') == 'stochastic_matrix':
            noise = config['noise']
            functions = [StochasticMatrixTridiagonalQuadraticFunction.from_tridiagonal_quadratic(f, generator, noise) 
                         for f in functions]
        else:
            meta = TridiagonalQuadraticOptimizationProblemMeta(functions)
    if config.get('random', True):
        point = generate_random_vector(config['dim'], generator)
    else:
        point = np.load(os.path.join(config['dump_path'], 'point.npy'))
    if config.get('scale_initial_point', None) is not None:
        point = config['scale_initial_point'] * point
    return functions, point, {}, meta


def prepare_mean_quadratic(path_to_dataset, config, generator):
    def calculate_gamma(l_minus, prob, A, B, l_plus_omega, l_plus_minus_omega):
        inv_gamma = l_minus + math.sqrt(((1 - prob) / prob) * ((A - B) * l_plus_omega ** 2 + B * l_plus_minus_omega ** 2))
        return 1 / inv_gamma
    
    meta = OptimizationProblemMeta()
    if config.get('sampling_name', 'original_page') == 'original_page':
        sampling_name = SamplingType.UNIFORM_WITH_REPLACEMENT
    else:
        sampling_name = config['sampling_name']
    functions = MeanTridiagonalQuadraticFunction.load(
        os.path.join(config['dump_path'], "functions"), seed=generator, 
        sampling=sampling_name)
    point = np.load(os.path.join(config['dump_path'], 'point.npy'))
    if config.get('scale_initial_point', None) is not None:
        point = config['scale_initial_point'] * point
    
    if config['algorithm_master_params']['gamma'] is None:
        quadratic_functions = functions.get_quadratic_functions()
        batch_size = config['algorithm_master_params']['batch_size']
        if config['sampling_name'] == SamplingType.UNIFORM_WITH_REPLACEMENT or config['sampling_name'] == 'original_page':
            A = B = 1 / batch_size
            if config['sampling_name'] == 'original_page':
                B = 0
            l_minus = TridiagonalQuadraticFunction.liptschitz_gradient_constant_functions(quadratic_functions)
            l_plus_omega = TridiagonalQuadraticFunction.liptschitz_gradient_constant_plus_functions(quadratic_functions)
            l_plus_minus_omega = TridiagonalQuadraticFunction.smoothness_variance_bound_functions(quadratic_functions)
            prob = batch_size / (batch_size + len(quadratic_functions))
        elif config['sampling_name'] == SamplingType.IMPORTANCE:
            A = B = 1 / batch_size
            l_minus = TridiagonalQuadraticFunction.liptschitz_gradient_constant_functions(quadratic_functions)
            l_list = [TridiagonalQuadraticFunction.liptschitz_gradient_constant(f) for f in quadratic_functions]
            weights = l_list / np.sum(l_list)
            l_plus_omega = TridiagonalQuadraticFunction.liptschitz_gradient_constant_plus_functions(quadratic_functions, weights)
            l_plus_minus_omega = TridiagonalQuadraticFunction.smoothness_variance_bound_functions(quadratic_functions, weights)
            prob = batch_size / (batch_size + len(quadratic_functions))
        else:
            print(config['sampling_name'])
            raise RuntimeError("Wrong sampling name: {}".format(config['sampling_name']))
        gamma = calculate_gamma(l_minus, prob, A, B, l_plus_omega, l_plus_minus_omega)
        print("gamma: {}, L_minus: {}, L_plus_omega: {}, L_plus_minus_omega: {}, A: {}, B: {}, prob: {}".format(
                gamma, l_minus, l_plus_omega, l_plus_minus_omega, A, B, prob))
        config['algorithm_master_params']['gamma'] = gamma
    functions_nodes = [functions for _ in range(config['num_nodes'])]
    return functions_nodes, point, {}, meta


def run_experiments(path_to_dataset, dumps_path, config, basename):
    generator = np.random.default_rng(seed=config.get('seed', 42))
    if config['task'] == 'libsvm':
        functions, point, function_stats, meta = prepare_libsvm(path_to_dataset, config, generator)
    elif config['task'] == 'cifar10':
        functions, point, function_stats, meta = prepare_cifar10(path_to_dataset, config, generator)
    elif config['task'] == 'quadratic':
        functions, point, function_stats, meta = prepare_quadratic(path_to_dataset, config, generator)
    elif config['task'] == 'mean_quadratic':
        functions, point, function_stats, meta = prepare_mean_quadratic(path_to_dataset, config, generator)
    else:
        raise RuntimeError()
    algorithm_master_params = config['algorithm_master_params']
    algorithm_node_params = config.get('algorithm_node_params', {})
    compressor_params = dict(config.get('compressor_params', {}))
    compressor_params['dim'] = len(point)
    compressor_master_params = dict(config.get('compressor_master_params', {}))
    compressor_master_params['dim'] = len(point)
    if config.get('compressor_name', None) == 'group_permutation':
        compressor_params['nodes_indices_splits'] = function_stats['nodes_indices_splits']
    print("Dim: {}".format(compressor_params['dim']))
    params = {'algorithm_name': config['algorithm_name'],
              'algorithm_master_params': algorithm_master_params,
              'algorithm_node_params': algorithm_node_params,
              'compressor_name': config.get('compressor_name', None),
              'compressor_params': compressor_params,
              'config': config,
              'point': point.tolist(),
              'function_stats': function_stats,
              'path_to_dataset': path_to_dataset}
    transport_type = config.get('transport_type', 'communication_transport')
    if transport_type == 'communication_transport':
        optimizer = get_comm_algorithm(
            functions, point, seed=generator,
            algorithm_name=config['algorithm_name'], 
            algorithm_master_params=algorithm_master_params, 
            algorithm_node_params=algorithm_node_params,
            meta=meta,
            compressor_name=config.get('compressor_name', None), 
            compressor_params=compressor_params,
            compressor_master_name=config.get('compressor_master_name', None), 
            compressor_master_params=compressor_master_params,
            multiple_master_compressors=config.get('multiple_master_compressors', False),
            parallel=config.get('parallel', False),
            shared_memory_size=config.get('shared_memory_size', 0),
            shared_memory_len=config.get('shared_memory_len', 1),
            number_of_processes=config.get('number_of_processes', 1))
    elif transport_type == 'asyncrounous_transport':
        optimizer = get_async_algorithm(
            functions, point, 
            delays=config['delays'],
            seed=generator,
            algorithm_name=config['algorithm_name'], 
            algorithm_master_params=algorithm_master_params, 
            algorithm_node_params=algorithm_node_params,
            meta=meta)
    elif transport_type == 'asyncrounous_transport_with_communication':
        optimizer = get_async_algorithm_with_communication(
            functions, point, 
            delays=config['delays'],
            seed=generator,
            algorithm_name=config['algorithm_name'], 
            algorithm_master_params=algorithm_master_params, 
            algorithm_node_params=algorithm_node_params,
            meta=meta)
    elif transport_type == 'asyncrounous_transport_with_graphs':
        optimizer = get_async_algorithm_with_graphs(
            functions, point, 
            delays=config['delays'],
            seed=generator,
            algorithm_name=config['algorithm_name'], 
            algorithm_master_params=algorithm_master_params, 
            algorithm_node_params=algorithm_node_params,
            meta=meta)
    else:
        assert False
    params['gamma'] = float(optimizer._gamma)
    if transport_type == 'communication_transport':
        optimizer_stat = OptimizerStat(optimizer, params)
    elif transport_type in ['asyncrounous_transport', 'asyncrounous_transport_with_communication', 'asyncrounous_transport_with_graphs']:
        optimizer_stat = OptimizerAsyncStat(optimizer, params)
    else:
        assert False
    for index_iteration in range(config['number_of_iterations']):
        print(index_iteration)
        t = time.time()
        ok = optimizer_stat.step()
        print("Time step: {}".format(time.time() - t))
        if not ok:
            break
        if index_iteration % config.get('save_rate', 1000) == 0:
            optimizer_stat.dump(dumps_path, basename)
    optimizer_stat.dump(dumps_path, basename)
    optimizer.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--path_to_dataset')

    args = parser.parse_args()
    basename = os.path.basename(args.config).split('.')[0]
    configs = yaml.safe_load(open(args.config))
    for config in configs:
        if config.get('parallel', False):
            torch.multiprocessing.set_sharing_strategy('file_system')
            if config.get('multiprocessing_spawn', False):
                multiprocessing.set_start_method("spawn")
        run_experiments(args.path_to_dataset, args.dumps_path, config, basename)


if __name__ == "__main__":
    main()