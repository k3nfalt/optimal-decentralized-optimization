import os
import argparse
import json
from posixpath import basename
import yaml
import math
import time
import multiprocessing
from io import BytesIO

import numpy as np
import torch


class OptimizerStat(object):
    def __init__(self, optimizer, params):
        self._optimizer = optimizer
        self._stat = {'times': []}
        self._params = params
        self._index = 0
        self._quality_check = False
        if self._params['config'].get('calculate_function', False):
            self._stat['function_values'] = []
        self._stat['number_of_iterations'] = 0

        
    def step(self):
        self._optimizer.step()
        with self._optimizer.ignore_statistics():
            if self._index % self._params['config'].get('quality_check_rate', 1) == 0:
                if self._params['config'].get('calculate_function', False):
                    function_value = float(self._optimizer.calculate_function())
                    self._stat['function_values'].append(function_value)
                    print("Function value: {}".format(function_value))
                self._stat['times'].append(self._optimizer.get_time())
                if self._params['config'].get('calculate_function', False):
                    if math.isnan(function_value):
                        return False
            self._index += 1
            self._stat['number_of_iterations'] = self._index
        return True
    
    def dump(self, path, name):
        dm = {'stat': self._stat, 'params': self._params}
        with self._optimizer.ignore_statistics():
            dm['point'] = dm['point'] = serialize_point(self._optimizer.get_point())
        with open(os.path.join(path, name), 'w') as fd:
            json.dump(dm, fd)


def mean(vectors):
    return sum(vectors) / float(len(vectors))


def run_experiments(path_to_dataset, dumps_path, config, basename):
    generator = np.random.default_rng(seed=config.get('seed', 42))
    if config['task'] == 'quadratic':
        functions, point, function_stats, meta = prepare_quadratic(path_to_dataset, config, generator)
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
              'config': config,
              'point': point.tolist(),
              'function_stats': function_stats,
              'path_to_dataset': path_to_dataset}
    optimizer = get_algorithm(functions, point, seed=generator,
                              algorithm_name=config['algorithm_name'], 
                              algorithm_master_params=algorithm_master_params, 
                              algorithm_node_params=algorithm_node_params,
                              meta=meta,
                              compressor_name=config.get('compressor_name', None), 
                              compressor_params=compressor_params,
                              compressor_master_name=config.get('compressor_master_name', None), 
                              compressor_master_params=compressor_master_params,
                              parallel=config.get('parallel', False),
                              shared_memory_size=config.get('shared_memory_size', 0),
                              shared_memory_len=config.get('shared_memory_len', 1),
                              number_of_processes=config.get('number_of_processes', 1))
    params['gamma'] = float(optimizer._gamma)
    optimizer_stat = OptimizerStat(optimizer, params)
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
