import tempfile
import os
import itertools
import numpy as np
import argparse
import copy
import yaml
import json

from distributed_optimization_library.function import QuadraticFunction, TridiagonalQuadraticFunction

ibex_template = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time=10:30:00
#SBATCH --mem=10G
#SBATCH --constraint=intel
#SBATCH --cpus-per-task={cpus_per_task}

cd /home/tyurina/rsync_watch/distributed_optimization_library/
module load singularity
singularity exec ~/Sync/Singularity.sif sh {singularity_script}
'''

singularity_template = 'PYTHONPATH=./code python3 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset ~/tmp --dumps_path ~/exepriments/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": "quadratic",
  "shared_memory_size": 100000,
  "random": False,
  "algorithm_master_params": {"gamma": None},
  "compressor_params": {},
  "calculate_norm_of_gradients": True
}


def generate_task(num_nodes, dim, noise_lambda, seed, strongly_convex_constant):
    print("#" * 100)
    print("Num nodes: {}, Noise lambda: {}".format(num_nodes, noise_lambda))
    functions = TridiagonalQuadraticFunction.create_worst_case_functions(
        num_nodes, dim, seed=seed, noise_lambda=noise_lambda, strongly_convex_constant=strongly_convex_constant)
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
    return functions, function_stats


def dump_task(folder, functions, function_stats):
    os.mkdir(folder)
    TridiagonalQuadraticFunction.dump_functions(functions, os.path.join(folder, "functions"))
    with open(os.path.join(folder, 'function_stats.json'), 'w') as fd:
        json.dump(function_stats, fd)
    point = np.zeros((function_stats['dim'],), dtype=np.float32)
    point[0] += np.sqrt(function_stats['dim'])
    np.save(os.path.join(folder, 'point'), point)


def generate_yaml(dim, noise_lambdas, num_nodes_list, experiments_name, strongly_convex_constant,
                  step_size_range, parallel, seed, top_k_factor, compressors,
                  ef21_init_with_gradients, cpus_per_task, number_of_processes):
    os.mkdir(os.path.abspath('/home/tyurina/exepriments/{experiments_name}'.format(experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('/home/tyurina/exepriments/{experiments_name}/source_folder'.format(experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    script_to_execute = ''
    generator = np.random.default_rng(seed=seed)
    folders = []
    problem_params = []
    algorithm_name = 'rand_diana'
    for index, (noise_lambda, num_nodes) in enumerate(itertools.product(noise_lambdas, num_nodes_list)):
        folder = os.path.join(tmp_dir, "problem_{}".format(index))
        functions, function_stats = generate_task(num_nodes, dim, noise_lambda, generator,
                                                  strongly_convex_constant)
        dump_task(folder, functions, function_stats)
        folders.append(folder)
        problem_params.append((noise_lambda, num_nodes))
    for index, (gamma_multiply, compressor_name, optimal_scale, folder) in enumerate(
            itertools.product([2**i for i in range(step_size_range[0], step_size_range[1])],
                              compressors,
                              [True, False],
                              folders)):
        with open(os.path.join(folder, 'function_stats.json')) as fd:
            function_stats = json.load(fd)
        noise_lambda = function_stats['noise_lambda'] 
        num_nodes = function_stats['num_nodes']
        yaml_prepared = copy.deepcopy(yaml_template)
        yaml_prepared['num_nodes'] = num_nodes
        yaml_prepared['dim'] = dim
        yaml_prepared['parallel'] = parallel
        if compressor_name == 'top_k':
            gamma_multiply *= 2**top_k_factor
        yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
        yaml_prepared['number_of_iterations'] = 20000
        yaml_prepared['homogeneous'] = False
        yaml_prepared['number_of_coordinates'] = max(int(dim / num_nodes), 1)
        yaml_prepared['noise_lambda'] = noise_lambda
        yaml_prepared['seed'] = int(generator.integers(10**9))
        yaml_prepared['compressor_name'] = compressor_name
        yaml_prepared['number_of_processes'] = min(num_nodes, number_of_processes)
        yaml_prepared['dump_path'] = folder
        
        if compressor_name == 'rand_k':
            yaml_prepared['algorithm_name'] = algorithm_name
            yaml_prepared['compressor_params']['number_of_coordinates'] = yaml_prepared['number_of_coordinates']
            yaml_prepared['algorithm_master_params']['optimal_scale'] = optimal_scale
        elif compressor_name == 'permutation' or compressor_name == 'permutation_fixed_blocks':
            yaml_prepared['algorithm_name'] = algorithm_name + "_permutation"
            yaml_prepared['compressor_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
        elif compressor_name == 'nodes_permutation':
            yaml_prepared['algorithm_name'] = algorithm_name + "_nodes_permutation"
            yaml_prepared['compressor_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
        elif compressor_name == 'top_k':
            yaml_prepared['algorithm_name'] = 'ef21'
            yaml_prepared['compressor_params']['number_of_coordinates'] = yaml_prepared['number_of_coordinates']
            yaml_prepared['number_of_iterations'] = 40000
            yaml_prepared['algorithm_master_params']['init_with_gradients'] = ef21_init_with_gradients
        else:
            raise RuntimeError()
        
        config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
        with open(config_name, 'w') as fd:
            yaml.dump([yaml_prepared], fd)
        singularity_script = singularity_template.format(
          experiments_name=experiments_name,
          config_name=config_name)
        singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
        open(singularity_script_name, 'w').write(singularity_script)
        ibex_str = ibex_template.format(singularity_script=singularity_script_name,
                                        cpus_per_task=cpus_per_task)
        script_name = os.path.join(tmp_dir, 'script_{}.sh'.format(index))
        open(script_name, 'w').write(ibex_str)
        script_to_execute += 'sbatch {}\n'.format(script_name)
    final_script_name = os.path.join(tmp_dir, 'execute.sh')
    open(final_script_name, 'w').write(script_to_execute)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', required=True, type=int)
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--noise_lambdas', required=True, nargs='+', type=float)
    parser.add_argument('--num_nodes_list', required=True, nargs='+', type=int)
    parser.add_argument('--strongly_convex_constant', type=float, default=1e-5)
    parser.add_argument('--step_size_range', required=True, nargs='+', type=int)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--top_k_factor', type=int, default=0)
    parser.add_argument('--compressors', nargs='+', default=['rand_k', 'permutation', 'top_k'])
    parser.add_argument('--ef21_init_with_gradients', action='store_true')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    args = parser.parse_args()
    
    generate_yaml(args.dim, args.noise_lambdas, args.num_nodes_list,
                  args.experiments_name, args.strongly_convex_constant,
                  args.step_size_range, args.parallel,
                  args.seed, args.top_k_factor, args.compressors, args.ef21_init_with_gradients,
                  args.cpus_per_task, args.number_of_processes)

if __name__ == "__main__":
    main()
