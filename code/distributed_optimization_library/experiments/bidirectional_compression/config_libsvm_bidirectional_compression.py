import itertools
import tempfile
import math
import os
import sys
import yaml
import copy
import json
import numpy as np
import argparse

from distributed_optimization_library.function import TridiagonalQuadraticFunction

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
singularity exec ~/rsync_watch/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif sh {singularity_script}
'''

singularity_template = 'MKL_NUM_THREADS=1 PYTHONPATH=./code python3.7 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset {dataset_path} --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": None,
  "algorithm_master_params": {"gamma": 1.0},
  "algorithm_node_params": {},
  "compressor_params": {},
  "compressor_master_params": {},
  "calculate_norm_of_gradients": True,
  "calculate_gradient_estimator_error": False,
  "calculate_function": True,
  "save_rate": 1000,
  "shared_memory_len": 2
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

def generate_yaml(dumps_path, dataset_path, experiments_name, dataset, num_nodes, step_size_range, number_of_seeds,
                  number_of_iterations, algorithm_names, func, parallel, cpus_per_task,
                  time, number_of_processes, compressor,
                  calculate_smoothness_variance,
                  scale_initial_point, reg_paramterer, ef21_init_with_gradients,
                  point_initializer, batch_size, homogeneous, number_of_coordinates, number_of_coordinates_master,
                  quality_check_rate, logistic_regression_nonconvex,
                  gb_per_task, oracle, mega_batch_size, no_initial_mega_batch_size,
                  no_initial_mega_batch_size_zero,
                  split_into_groups_by_labels, equalize_to_same_number_samples_per_class,
                  subsample_classes, fixed_seed, calculate_accuracy, not_only_intel, acc_sum_gamma_inverse_omega,
                  add_strongly_convex_constant, task, quad_strongly_convex_constant, quad_dim, quad_noise_lambda,
                  acc_scale_prob):
    folder_with_project = os.getcwd()
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=dumps_path, experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=dumps_path, experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    script_to_execute = ''
    generator = np.random.default_rng(seed=42)
    index = 0
    
    if task == 'quadratic':
        # for index, (noise_lambda, num_nodes) in enumerate(itertools.product(noise_lambdas, num_nodes_list)):
        folder = os.path.join(tmp_dir, "problem_{}".format(index))
        generate_type = 'worst_case'
        functions, function_stats = generate_task(num_nodes, quad_dim, quad_noise_lambda, generator,
                                                  quad_strongly_convex_constant,
                                                  generate_type)
        dump_task(folder, functions, function_stats)
    
    for _, (algorithm_name, seed_number) in enumerate(
        itertools.product(algorithm_names, range(number_of_seeds))):
        for gamma_multiply in [2**i for i in range(step_size_range[0], step_size_range[1])]:
            yaml_prepared = copy.deepcopy(yaml_template)
            yaml_prepared['task'] = task
            if yaml_prepared['task'] == 'libsvm':
                yaml_prepared['dataset_name'] = dataset
            elif yaml_prepared['task'] == 'quadratic':
                yaml_prepared['dump_path'] = folder
                yaml_prepared['dim'] = int(quad_dim)
                yaml_prepared['random'] = False
            else:
                assert False
            yaml_prepared['num_nodes'] = num_nodes
            yaml_prepared['algorithm_name'] = algorithm_name
            yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
            if algorithm_name == 'accelerated_bidirectional_diana':
                yaml_prepared['algorithm_master_params']['sum_gamma_inverse_omega'] = acc_sum_gamma_inverse_omega
                yaml_prepared['algorithm_master_params']['add_strongly_convex_constant'] = add_strongly_convex_constant
                yaml_prepared['algorithm_master_params']['scale_prob'] = acc_scale_prob
            if oracle == 'gradient':
                if compressor == 'top_k':
                    assert number_of_coordinates_master == 1
                    if algorithm_name == 'mcm':
                        yaml_prepared['compressor_name'] = 'unbiased_top_k'
                        yaml_prepared['compressor_master_name'] = 'unbiased_top_k'
                    elif algorithm_name == 'ef21_bc':
                        yaml_prepared['compressor_name'] = 'top_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = 1
                        yaml_prepared['compressor_master_name'] = 'top_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = 1
                    elif algorithm_name == 'dcgd_ef21_primal':
                        yaml_prepared['compressor_name'] = 'unbiased_top_k'
                        yaml_prepared['compressor_master_name'] = 'top_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = 1
                    elif algorithm_name in ['zero_marina_ef21_primal', 'diana_ef21_primal']:
                        yaml_prepared['compressor_name'] = 'unbiased_top_k'
                        yaml_prepared['compressor_master_name'] = 'top_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = 1
                    else:
                        assert False
                if compressor == 'rand_k':
                    number_of_coordinates_master = number_of_coordinates if number_of_coordinates_master is None else number_of_coordinates_master
                    if algorithm_name == 'mcm':
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'rand_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name == 'ef21_bc':
                        yaml_prepared['compressor_name'] = 'biased_rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'biased_rand_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name == 'dcgd_ef21_primal':
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'biased_rand_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name in ['zero_marina_ef21_primal', 'diana_ef21_primal']:
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'biased_rand_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name in ['zero_marina', 'marina', 'diana']:
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                    else:
                        assert False
                if compressor == 'rand_k_identity':
                    if algorithm_name == 'mcm':
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'identity_unbiased'
                    elif algorithm_name == 'ef21_bc':
                        yaml_prepared['compressor_name'] = 'biased_rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'identity_biased'
                    elif algorithm_name == 'dcgd_ef21_primal':
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'identity_biased'
                    elif algorithm_name in ['zero_marina_ef21_primal', 'diana_ef21_primal', 'accelerated_bidirectional_diana']:
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'identity_biased'
                    elif algorithm_name in ['zero_marina', 'marina', 'diana']:
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                    else:
                        assert False
                if compressor == 'identity':
                    if algorithm_name in ['zero_marina_ef21_primal', 'diana_ef21_primal', 'accelerated_bidirectional_diana']:
                        yaml_prepared['compressor_name'] = 'identity_unbiased'
                        yaml_prepared['compressor_master_name'] = 'identity_biased'
                    else:
                        assert False
                if compressor == 'rand_k_top_k':
                    number_of_coordinates_master = 1 if number_of_coordinates_master is None else number_of_coordinates_master
                    if algorithm_name == 'mcm':
                        assert number_of_coordinates_master == 1
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'unbiased_top_k'
                    elif algorithm_name == 'ef21_bc':
                        yaml_prepared['compressor_name'] = 'biased_rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'top_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name == 'dcgd_ef21_primal':
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'top_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name in ['zero_marina_ef21_primal', 'diana_ef21_primal', 'accelerated_bidirectional_diana']:
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
                        yaml_prepared['compressor_master_name'] = 'top_k'
                        yaml_prepared['compressor_master_params']['number_of_coordinates'] = number_of_coordinates_master
                    elif algorithm_name in ['zero_marina', 'marina', 'diana']:
                        yaml_prepared['compressor_name'] = 'rand_k'
                        yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
            yaml_prepared['number_of_iterations'] = number_of_iterations
            if algorithm_name == 'accelerated_bidirectional_diana':
                yaml_prepared['number_of_iterations'] //= 2
            yaml_prepared['shuffle'] = True
            yaml_prepared['split_into_groups_by_labels'] = split_into_groups_by_labels
            yaml_prepared['equalize_to_same_number_samples_per_class'] = equalize_to_same_number_samples_per_class
            yaml_prepared['subsample_classes'] = subsample_classes
            yaml_prepared['ignore_remainder'] = True
            yaml_prepared['calculate_accuracy'] = calculate_accuracy
            yaml_prepared['homogeneous'] = homogeneous
            yaml_prepared['shared_memory_size'] = 100000
            yaml_prepared['parallel'] = parallel
            yaml_prepared['number_of_processes'] = number_of_processes
            yaml_prepared['function'] = func
            if func in ['logistic_regression', 'stochastic_logistic_regression'] and logistic_regression_nonconvex > 0.0:
                yaml_prepared['function_parameters'] = {'nonconvex_regularizer': True,
                                                        'reg_paramterer': logistic_regression_nonconvex}
            yaml_prepared['seed'] = seed_number if fixed_seed is None else fixed_seed
            yaml_prepared['quality_check_rate'] = quality_check_rate
            if scale_initial_point is not None:
                assert scale_initial_point > 0.0
                yaml_prepared['scale_initial_point'] = scale_initial_point
            if point_initializer is not None:
                yaml_prepared['point_initializer'] = point_initializer
            if reg_paramterer is not None and reg_paramterer > 0.0:
                yaml_prepared['reg_paramterer'] = reg_paramterer
            
            config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
            with open(config_name, 'w') as fd:
                yaml.dump([yaml_prepared], fd)
            singularity_script = singularity_template.format(experiments_name=experiments_name,
                                                             config_name=config_name,
                                                             dumps_path=dumps_path,
                                                             dataset_path=dataset_path)
            singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
            open(singularity_script_name, 'w').write(singularity_script)
            if not not_only_intel:
                contraint = "#SBATCH --constraint=intel"
            else:
                contraint = ""
            ibex_str = ibex_template.format(folder_with_project=folder_with_project,
                                            singularity_script=singularity_script_name,
                                            experiments_name=experiments_name,
                                            cpus_per_task=cpus_per_task,
                                            time=time,
                                            gb_per_task=gb_per_task,
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
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--num_nodes_list', required=True, nargs='+', type=int)
    parser.add_argument('--step_size_range', required=True, nargs='+', type=int)
    parser.add_argument('--number_of_seeds', type=int, default=1)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--algorithm_names', required=True, nargs='+')
    parser.add_argument('--function', default="logistic_regression")
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--gb_per_task', default=30, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    parser.add_argument('--time', default=10, type=int)
    parser.add_argument('--compressor', default='top_k')
    parser.add_argument('--calculate_smoothness_variance', action='store_true')
    parser.add_argument('--point_initializer')
    parser.add_argument('--scale_initial_point', type=float)
    parser.add_argument('--reg_paramterer', type=float)
    parser.add_argument('--ef21_init_with_gradients', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--mega_batch_size', default=128, type=int)
    parser.add_argument('--number_of_coordinates', default=1, type=int)
    parser.add_argument('--number_of_coordinates_master', default=None, type=int)
    parser.add_argument('--homogeneous', action='store_true')
    parser.add_argument('--logistic_regression_nonconvex', default=0.0, type=float)
    parser.add_argument('--quality_check_rate', default=1, type=int)
    parser.add_argument('--oracle', default='gradient')
    parser.add_argument('--no_initial_mega_batch_size', action='store_true')
    parser.add_argument('--no_initial_mega_batch_size_zero', action='store_true')
    parser.add_argument('--split_into_groups_by_labels', action='store_true')
    parser.add_argument('--equalize_to_same_number_samples_per_class', type=int)
    parser.add_argument('--subsample_classes', type=int)
    parser.add_argument('--fixed_seed', type=int)
    parser.add_argument('--calculate_accuracy', action='store_true')
    parser.add_argument('--not_only_intel', action='store_true')
    parser.add_argument('--acc_sum_gamma_inverse_omega', action='store_true')
    parser.add_argument('--add_strongly_convex_constant', type=float, default=0.0)
    parser.add_argument('--acc_scale_prob', type=float, default=1.0)
    parser.add_argument('--quad_strongly_convex_constant', type=float, default=0.0)
    parser.add_argument('--task', default="libsvm")
    parser.add_argument('--quad_dim', type=int, default=None)
    parser.add_argument('--quad_noise_lambda', type=float, default=0.0)
    args = parser.parse_args()
    assert len(args.step_size_range) == 2
    step_size_range = args.step_size_range
    generate_yaml(args.dumps_path, args.dataset_path, args.experiments_name, args.dataset, args.num_nodes_list[0], args.step_size_range,
                  args.number_of_seeds, args.number_of_iterations, args.algorithm_names,
                  args.function, args.parallel, args.cpus_per_task, args.time,
                  args.number_of_processes, args.compressor,
                  args.calculate_smoothness_variance,
                  args.scale_initial_point, args.reg_paramterer,
                  args.ef21_init_with_gradients, args.point_initializer,
                  args.batch_size, args.homogeneous, args.number_of_coordinates, args.number_of_coordinates_master,
                  args.quality_check_rate, args.logistic_regression_nonconvex,
                  args.gb_per_task, args.oracle, args.mega_batch_size,
                  args.no_initial_mega_batch_size, args.no_initial_mega_batch_size_zero,
                  args.split_into_groups_by_labels, args.equalize_to_same_number_samples_per_class,
                  args.subsample_classes, args.fixed_seed,
                  args.calculate_accuracy, args.not_only_intel,
                  args.acc_sum_gamma_inverse_omega, args.add_strongly_convex_constant,
                  args.task, args.quad_strongly_convex_constant, args.quad_dim, args.quad_noise_lambda,
                  args.acc_scale_prob)
