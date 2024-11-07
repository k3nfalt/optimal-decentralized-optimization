import os
import yaml
import multiprocessing
import argparse
import time
import numpy as np
import random
import json

import torch


from distributed_optimization_library.models.toy_models import prepare_toy_model
from distributed_optimization_library.function import parameters_to_tensor, tensor_to_parameters, BaseStochasticFunction

from distributed_optimization_library.experiments.local_optimization_pytorch.optimizers import GradientDescent, AdaptiveGradientDescent

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


class StochasticToyModel():
    def __init__(self, model):
        self._model = model
        self._model.train()

    def gradient(self, point):
        tensor_to_parameters(self._model.parameters(), point)
        loss, gradient = self._gradient()
        self._last_loss = loss
        self._last_accuracy = None
        return gradient

    def parameters(self):
        return self._model.parameters()
    
    def current_point(self):
        return parameters_to_tensor(self._model.parameters())
    
    def last_loss_and_accuracy(self):
        return self._last_loss, self._last_accuracy
    
    def _gradient(self):
        loss = self._model()
        self._model.zero_grad()
        loss.backward()
        gradient = parameters_to_tensor(self._model.parameters(), grad=True)
        return loss, gradient


def run_experiments(dumps_path, config, basename):
    if config.get('use_double', False):
        torch.set_default_tensor_type(torch.DoubleTensor)
    dump_path = os.path.join(dumps_path, basename)
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model = prepare_toy_model(model_type=config['model_type'], **config['model_params'])
        
    model_wrapper = StochasticToyModel(model)
    point = model_wrapper.current_point()
    if config['optimizer'] == 'gd':
        optimizer_wrapper = GradientDescent(model_wrapper, point, lr=config['learning_rate'])
    elif config['optimizer'] == 'adaptive_stable_gd':
        optimizer_wrapper = AdaptiveGradientDescent(model_wrapper, point, lr=10**9, stable=True)
    stat = {'batch_loss': [], 'batch_accuracy': [], 'batch_norm_of_gradient': [],
            'estimate_sigma_2': [], 'estimate_norm_of_gradient': [],
            'norm_of_gradient': [], 'sigma_2': [],
            'batch_diff_gradient': [], 'batch_diff_points': [],
            'batch_dot_gradient': [], 'learning_rates': [], 'extra_stat': []}
    dump = {'config': config, 'stat': stat}
    batch_index = 0
    while True:
        start_time = time.time()
        optimizer_wrapper.step()
        iter_time = time.time() - start_time
        loss, accuracy = model_wrapper.last_loss_and_accuracy()
        stat['batch_loss'].append(loss.item())
        stat['batch_accuracy'].append(accuracy)
        stat['learning_rates'].append(optimizer_wrapper._learning_rate)
        stat['extra_stat'].append(optimizer_wrapper.get_stat())
        print("iter: {iter} / {number_of_batches} loss: {loss}, acc: {accuracy}, time: {iter_time}".format(
              loss=loss, iter_time=iter_time, accuracy=accuracy,
              iter=batch_index, number_of_batches=config['number_of_batches']))
        if batch_index >= config['number_of_batches']:
            save_model(dump_path, model, batch_index)
            save_dump(dump_path, dump)
            break
        if config.get('save_model_every', None) is not None and batch_index % config['save_model_every'] == 0:
            save_model(dump_path, model, batch_index)
        if batch_index % config.get('save_every', 100) == 0:
            save_dump(dump_path, dump)
        batch_index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--config', required=True)

    args = parser.parse_args()
    basename = os.path.basename(args.config).split('.')[0]
    configs = yaml.safe_load(open(args.config))
    assert len(configs) == 1
    config = configs[0]
    if config.get('parallel', False):
        torch.multiprocessing.set_sharing_strategy('file_system')
        if config.get('multiprocessing_spawn', False):
            multiprocessing.set_start_method("spawn")
    run_experiments(args.dumps_path, config, basename)

if __name__ == "__main__":
    main()
