import math
import os
import itertools
import numpy as np
import argparse
import copy
import yaml
import json

from distributed_optimization_library.function import QuadraticFunction, TridiagonalQuadraticFunction
from distributed_optimization_library.function import SamplingType

ibex_template = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time=1:00:00
#SBATCH --mem=10G
#SBATCH --constraint=intel
#SBATCH --cpus-per-task={cpus_per_task}

cd {folder_with_project}
module load singularity
singularity exec ~/Sync/Singularity.sif sh {singularity_script}
'''

singularity_template = 'PYTHONPATH=./code python3 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset {dataset_path} --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": "mean_quadratic",
  "shared_memory_size": 100000,
  "random": False,
  "algorithm_master_params": {"gamma": None},
  "compressor_params": {},
  "calculate_norm_of_gradients": True,
  "statistics": True
}


def generate_task(num_nodes, dim, noise_lambda, seed, strongly_convex_constant,generate_type):
    print("#" * 100)
    print("Num nodes: {}, Noise lambda: {}".format(num_nodes, noise_lambda))
    if generate_type == 'lipt_different':
        functions = TridiagonalQuadraticFunction.create_convex_different_liptschitz(
            num_nodes, dim, seed=seed, noise_lambda=noise_lambda)
    elif generate_type == 'worst_case':
        functions = TridiagonalQuadraticFunction.create_worst_case_functions(
            num_nodes, dim, seed=seed, noise_lambda=noise_lambda, 
            strongly_convex_constant=strongly_convex_constant)
    min_eigenvalue = TridiagonalQuadraticFunction.min_eigenvalue_functions(functions)
    assert min_eigenvalue > -1e-4, min_eigenvalue
    print("Min eigenvalue: {}".format(min_eigenvalue))
    function_stats = {}
    smoothness_variance_bound = TridiagonalQuadraticFunction.smoothness_variance_bound_functions(functions)
    print("Smoothness variance bound: {}".format(smoothness_variance_bound))
    liptschitz_gradient_constant = TridiagonalQuadraticFunction.liptschitz_gradient_constant_functions(functions)
    print("Liptschitz gradient constant: {}".format(liptschitz_gradient_constant))
    analytical_solution = TridiagonalQuadraticFunction.analytical_solution_functions(functions)
    min_value = TridiagonalQuadraticFunction.value_functions(functions, analytical_solution)
    print("Min value: {}".format(min_value))
    function_stats['smoothness_variance_bound'] = float(smoothness_variance_bound)
    function_stats['liptschitz_gradient_constant'] = float(liptschitz_gradient_constant)
    function_stats['min_eigenvalue'] = float(min_eigenvalue)
    function_stats['noise_lambda'] = float(noise_lambda)
    function_stats['dim'] = int(dim)
    function_stats['num_nodes'] = int(num_nodes)
    function_stats['min_value'] = float(min_value)
    liptschitz_gradient_constant_plus = TridiagonalQuadraticFunction.liptschitz_gradient_constant_plus_functions(functions)
    function_stats['liptschitz_gradient_constant_plus'] = liptschitz_gradient_constant_plus
    local_liptschitz_constants = [float(f.liptschitz_gradient_constant()) for f in functions]
    function_stats['local_liptschitz_gradient_constants'] = local_liptschitz_constants
    return functions, function_stats


def dump_task(folder, functions, function_stats):
    os.mkdir(folder)
    TridiagonalQuadraticFunction.dump_functions(functions, os.path.join(folder, "functions"))
    with open(os.path.join(folder, 'function_stats.json'), 'w') as fd:
        json.dump(function_stats, fd)
    point = np.zeros((function_stats['dim'],), dtype=np.float32)
    point[0] += np.sqrt(function_stats['dim'])
    np.save(os.path.join(folder, 'point'), point)


def generate_yaml(dataset_path, dumps_path, dim, noise_lambdas, num_nodes_list, experiments_name, strongly_convex_constant,
                  algorithm_name, theretical_step_size, step_size_range, parallel, seed, top_k_factor, samplings,
                  cpus_per_task, number_of_processes, number_of_iterations, batch_sizes,
                  generate_type):
    folder_with_project = os.getcwd()
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=dumps_path, experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=dumps_path, experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    script_to_execute = ''
    generator = np.random.default_rng(seed=seed)
    folders = []
    problem_params = []
    compressor_name = 'identity_unbiased'
    if theretical_step_size:
        step_sizes = [None]
    else:
        step_sizes = [2**i for i in range(step_size_range[0], step_size_range[1])]
    for index, (noise_lambda, num_nodes) in enumerate(itertools.product(noise_lambdas, num_nodes_list)):
        folder = os.path.join(tmp_dir, "problem_{}".format(index))
        functions, function_stats = generate_task(num_nodes, dim, noise_lambda, generator,
                                                  strongly_convex_constant,
                                                  generate_type)
        dump_task(folder, functions, function_stats)
        folders.append(folder)
        problem_params.append((noise_lambda, num_nodes))
    for index, (gamma_multiply, sampling_name, folder, batch_size) in enumerate(
            itertools.product(step_sizes,
                              samplings,
                              folders,
                              batch_sizes)):
        with open(os.path.join(folder, 'function_stats.json')) as fd:
            function_stats = json.load(fd)
        noise_lambda = function_stats['noise_lambda'] 
        num_nodes = function_stats['num_nodes']
        yaml_prepared = copy.deepcopy(yaml_template)
        yaml_prepared['num_nodes'] = num_nodes
        yaml_prepared['dim'] = dim
        yaml_prepared['parallel'] = parallel
        yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
        yaml_prepared['algorithm_master_params']['batch_size'] = batch_size
        yaml_prepared['number_of_iterations'] = number_of_iterations
        yaml_prepared['homogeneous'] = False
        yaml_prepared['number_of_coordinates'] = max(int(dim / num_nodes), 1)
        yaml_prepared['noise_lambda'] = noise_lambda
        yaml_prepared['seed'] = int(generator.integers(10**9))
        yaml_prepared['compressor_name'] = compressor_name
        yaml_prepared['number_of_processes'] = min(num_nodes, number_of_processes)
        yaml_prepared['dump_path'] = folder
        yaml_prepared['sampling_name'] = sampling_name
        
        assert compressor_name == 'identity_unbiased'
        yaml_prepared['algorithm_name'] = algorithm_name
        
        config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
        with open(config_name, 'w') as fd:
            yaml.dump([yaml_prepared], fd)
        singularity_script = singularity_template.format(
          dataset_path=dataset_path,
          dumps_path=dumps_path,
          experiments_name=experiments_name,
          config_name=config_name)
        singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
        open(singularity_script_name, 'w').write(singularity_script)
        ibex_str = ibex_template.format(folder_with_project=folder_with_project,
                                        singularity_script=singularity_script_name,
                                        cpus_per_task=cpus_per_task)
        script_name = os.path.join(tmp_dir, 'script_{}.sh'.format(index))
        open(script_name, 'w').write(ibex_str)
        script_to_execute += 'sbatch {}\n'.format(script_name)
    final_script_name = os.path.join(tmp_dir, 'execute.sh')
    open(final_script_name, 'w').write(script_to_execute)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--dim', required=True, type=int)
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--noise_lambdas', required=True, nargs='+', type=float)
    parser.add_argument('--num_nodes_list', required=True, nargs='+', type=int)
    parser.add_argument('--strongly_convex_constant', type=float, default=1e-5)
    parser.add_argument('--algorithm_name', default="vr_marina")
    parser.add_argument('--theretical_step_size', action='store_true')
    parser.add_argument('--step_size_range', required=True, nargs='+', type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--top_k_factor', type=int, default=0)
    parser.add_argument('--samplings', nargs='+')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    parser.add_argument('--number_of_iterations', type=int, default=0)
    parser.add_argument('--batch_size', type=int, nargs='+', default=[1])
    parser.add_argument('--generate_type', default='lipt_different')
    args = parser.parse_args()
    
    generate_yaml(args.dataset_path, args.dumps_path, args.dim, args.noise_lambdas, args.num_nodes_list,
                  args.experiments_name, args.strongly_convex_constant,
                  args.algorithm_name, args.theretical_step_size, 
                  args.step_size_range, args.parallel,
                  args.seed, args.top_k_factor, args.samplings,
                  args.cpus_per_task, args.number_of_processes,
                  args.number_of_iterations, args.batch_size,
                  args.generate_type)

if __name__ == "__main__":
    main()
