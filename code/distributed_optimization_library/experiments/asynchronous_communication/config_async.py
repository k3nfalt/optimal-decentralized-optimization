import itertools
import os
import sys
import yaml
import copy
import json
import math
import numpy as np
import argparse

from distributed_optimization_library.function import TridiagonalQuadraticFunction
from distributed_optimization_library.asynchronous.algorithm_with_communication import METHOD_GRADIENT, METHOD_SEND

ibex_template = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob{experiments_name}
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time={time}:30:00
#SBATCH --mem={gb_per_task}G
#SBATCH --cpus-per-task={cpus_per_task}
{only_intel}

cd {folder_with_project}
module load singularity
singularity exec --bind /ibex ~/rsync_watch/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif sh {singularity_script}
'''

singularity_template = '{mkl_fix} PYTHONPATH=./code python3.7 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset {dataset_path} --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": None,
  "transport_type": "asyncrounous_transport_with_communication",
  "algorithm_master_params": {"gamma": 1.0},
  "algorithm_node_params": {},
  "calculate_function": True,
  "save_rate": 1000,
  "shared_memory_len": 2
}

def generate_task(num_nodes, dim, noise_lambda, seed, strongly_convex_constant, generate_type):
    print("#" * 100)
    print("Num nodes: {}, Noise lambda: {}".format(num_nodes, noise_lambda))
    if generate_type == 'worst_case':
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

def generate_yaml(args):
    folder_with_project = os.getcwd()
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=args.dumps_path, experiments_name=args.experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=args.dumps_path, experiments_name=args.experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    script_to_execute = ''
    generator = np.random.default_rng(seed=42)
    index = 0
    
    if args.task == 'quadratic':
        folder = os.path.join(tmp_dir, "problem_{}".format(index))
        generate_type = 'worst_case'
        functions, function_stats = generate_task(args.num_nodes, args.quad_dim, args.quad_noise_lambda, generator,
                                                  args.quad_strongly_convex_constant,
                                                  generate_type)
        dump_task(folder, functions, function_stats)
        dim = args.quad_dim
    else:
        # Dataset Mnist
        dim = 7850
    
    if args.delays == 'equal':
        assert False
        delays = [1] * args.num_nodes
    elif args.delays == 'only_equal_computation':
        delays = [1] * args.num_nodes
        delays = {METHOD_GRADIENT: [1 for i in range(args.num_nodes)],
                  METHOD_SEND: [0 for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt':
        delays = {METHOD_GRADIENT: [math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / dim for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_grad_sqrt_send_div_sqrt_dim':
        delays = {METHOD_GRADIENT: [math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / math.sqrt(dim) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_grad_sqrt_send_div_div_3_over_4_dim':
        delays = {METHOD_GRADIENT: [math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_grad_one_div_sqrt_dim_send':
        delays = {METHOD_GRADIENT: [math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [1 / math.sqrt(dim) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_grad_one_div_3_over_4_dim_send':
        delays = {METHOD_GRADIENT: [math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [1 / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_grad_one_div_dim_send':
        delays = {METHOD_GRADIENT: [math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [1 / dim for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_4_grad_one_div_sqrt_dim_send':
        delays = {METHOD_GRADIENT: [math.pow((i + 1), 1 / 4) for i in range(args.num_nodes)],
                  METHOD_SEND: [1 / math.sqrt(dim) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_4_grad_one_div_3_over_4_dim_send':
        delays = {METHOD_GRADIENT: [math.pow((i + 1), 1 / 4) for i in range(args.num_nodes)],
                  METHOD_SEND: [1 / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_4_grad_sqrt_send_div_3_over_4_dim':
        delays = {METHOD_GRADIENT: [math.pow((i + 1), 1 / 4) for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'sqrt_8_grad_sqrt_send_div_3_over_4_dim':
        delays = {METHOD_GRADIENT: [math.pow((i + 1), 1 / 8) for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'one_grad_sqrt_send_div_3_over_4_dim':
        delays = {METHOD_GRADIENT: [1 for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'inv_sqrt_grad_sqrt_send_div_3_over_4_dim':
        delays = {METHOD_GRADIENT: [1 / math.sqrt((i + 1)) for i in range(args.num_nodes)],
                  METHOD_SEND: [math.sqrt((i + 1)) / math.pow(dim, 0.75) for i in range(args.num_nodes)]}
    elif args.delays == 'one_grad_one_div_sqrt_dim_send':
        delays = {METHOD_GRADIENT: [1 for i in range(args.num_nodes)],
                  METHOD_SEND: [1 / math.sqrt(dim) for i in range(args.num_nodes)]}
    elif args.delays == 'gamma':
        assert False
        k, theta = 2, 200
        delays = []
        for _ in range(args.num_nodes):
            delay = 0
            while delay < 1.0:
                delay = np.random.gamma(k, theta) / (k * theta)
            delays.append(delay)
        delays = np.sort(delays).tolist()
    else:
        assert False

    for _, (algorithm_name, seed_number) in enumerate(
        itertools.product(args.algorithm_names, range(args.number_of_seeds))):
        for gamma_multiply in [2**i for i in range(step_size_range[0], step_size_range[1])]:
            batch_size_list = [None]
            if algorithm_name == 'shadowheart_sgd':
                batch_size_list = args.shadowheart_batch_size_list
            for batch_size in batch_size_list:
                yaml_prepared = copy.deepcopy(yaml_template)
                yaml_prepared['dim'] = dim
                if algorithm_name == 'shadowheart_sgd':
                    yaml_prepared['algorithm_master_params']['sigma_eps'] = batch_size
                    yaml_prepared['algorithm_master_params']['clip_number_of_communications'] = True
                if algorithm_name in ['shadowheart_sgd', 'qsgd']:
                    yaml_prepared['algorithm_master_params']['number_of_coordinates'] = args.number_of_coordinates
                yaml_prepared['task'] = args.task
                if yaml_prepared['task'] == 'quadratic':
                    yaml_prepared['dump_path'] = folder
                    yaml_prepared['dim'] = int(args.quad_dim)
                    yaml_prepared['random'] = False
                    yaml_prepared['type'] = 'stochastic'
                    yaml_prepared['noise'] = args.quad_noise
                    yaml_prepared['type_noise'] = args.quad_type_noise
                elif yaml_prepared['task'] == 'libsvm':
                    yaml_prepared['dataset_name'] = 'mnist'
                    if args.equalize_to_same_number_samples_per_class:
                        yaml_prepared['equalize_to_same_number_samples_per_class'] = args.equalize_to_same_number_samples_per_class
                    yaml_prepared['function'] = 'stochastic_logistic_regression'
                    yaml_prepared['batch_size'] = args.batch_size
                    yaml_prepared['num_nodes'] = args.num_nodes
                    yaml_prepared['homogeneous'] = True
                    yaml_prepared['shuffle'] = True
                else:
                    assert False
                yaml_prepared['delays'] = delays
                yaml_prepared['algorithm_name'] = algorithm_name
                yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
                yaml_prepared['number_of_iterations'] = args.number_of_iterations
                yaml_prepared['quality_check_rate'] = args.quality_check_rate
                if algorithm_name in ['asynchronous_sgd', 'fastest_sgd'] and args.scale_quality_check_rate is not None:
                    yaml_prepared['quality_check_rate'] *= args.scale_quality_check_rate
                assert yaml_prepared['quality_check_rate'] >= 1
                if algorithm_name in ['asynchronous_sgd', 'fastest_sgd']:
                    yaml_prepared['number_of_iterations'] = yaml_prepared['number_of_iterations'] * args.num_nodes
                if algorithm_name == 'shadowheart_sgd':
                    yaml_prepared['number_of_iterations'] = yaml_prepared['number_of_iterations'] * args.scale_num_iterations_shadowheart_sgd
                yaml_prepared['shuffle'] = True
                yaml_prepared['shared_memory_size'] = 100000
                yaml_prepared['seed'] = seed_number if args.fixed_seed is None else args.fixed_seed
                
                config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
                with open(config_name, 'w') as fd:
                    yaml.dump([yaml_prepared], fd, default_flow_style=None)
                if args.mkl_fix:
                    mkl_fix = "MKL_NUM_THREADS=1"
                else:
                    mkl_fix = ""
                singularity_script = singularity_template.format(experiments_name=args.experiments_name,
                                                                config_name=config_name,
                                                                dumps_path=args.dumps_path,
                                                                dataset_path=args.dataset_path,
                                                                mkl_fix=mkl_fix)
                singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
                open(singularity_script_name, 'w').write(singularity_script)
                if not args.not_only_intel:
                    contraint = "#SBATCH --constraint=intel"
                else:
                    contraint = ""
                ibex_str = ibex_template.format(folder_with_project=folder_with_project,
                                                singularity_script=singularity_script_name,
                                                experiments_name=args.experiments_name,
                                                cpus_per_task=args.cpus_per_task,
                                                time=args.time,
                                                gb_per_task=args.gb_per_task,
                                                only_intel=contraint)
                script_name = os.path.join(tmp_dir, 'script_{}.sh'.format(index))
                open(script_name, 'w').write(ibex_str)
                script_to_execute += 'sbatch {}\n'.format(script_name)
                index += 1
    final_script_name = os.path.join(tmp_dir, 'execute.sh')
    open(final_script_name, 'w').write(script_to_execute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--num_nodes', required=True, type=int)
    parser.add_argument('--step_size_range', required=True, nargs='+', type=int)
    parser.add_argument('--number_of_seeds', type=int, default=1)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--algorithm_names', required=True, nargs='+')
    parser.add_argument('--gb_per_task', default=30, type=int)
    parser.add_argument('--cpus_per_task', default=2, type=int)
    parser.add_argument('--time', default=10, type=int)
    parser.add_argument('--quality_check_rate', default=1, type=int)
    parser.add_argument('--fixed_seed', type=int)
    parser.add_argument('--not_only_intel', action='store_true')
    parser.add_argument('--task', default="quadratic")
    parser.add_argument('--quad_strongly_convex_constant', type=float, default=0.0)
    parser.add_argument('--quad_dim', type=int, default=None)
    parser.add_argument('--quad_noise_lambda', type=float, default=0.0)
    parser.add_argument('--quad_noise', type=float, default=0.0)
    parser.add_argument('--quad_type_noise', default='add')
    parser.add_argument('--delays', default='equal')
    parser.add_argument('--shadowheart_batch_size_list', nargs='+', type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--number_of_coordinates', default=None, type=int)
    parser.add_argument('--mkl_fix', action='store_true')
    parser.add_argument('--scale_quality_check_rate', default=None, type=int)
    parser.add_argument('--scale_num_iterations_shadowheart_sgd', default=1, type=int)
    parser.add_argument('--libsvm_function', default='stochastic_logistic_regression')
    parser.add_argument('--equalize_to_same_number_samples_per_class', default=None, type=int)
    args = parser.parse_args()
    assert len(args.step_size_range) == 2
    step_size_range = args.step_size_range
    generate_yaml(args)
