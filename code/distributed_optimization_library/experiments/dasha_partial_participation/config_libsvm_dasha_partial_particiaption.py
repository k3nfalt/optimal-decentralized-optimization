import itertools
import tempfile
import math
import os
import sys
import yaml
import copy
import numpy as np
import argparse

ibex_template = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob{experiments_name}
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time={time}:30:00
#SBATCH --mem={gb_per_task}G
#SBATCH --constraint=intel
#SBATCH --cpus-per-task={cpus_per_task}

cd {folder_with_project}
module load singularity
singularity exec ~/rsync_watch/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif sh {singularity_script}
'''

singularity_template = 'MKL_NUM_THREADS=1 PYTHONPATH=./code python3.7 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset {dataset_path} --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": "libsvm",
  "algorithm_master_params": {"gamma": 1.0},
  "algorithm_node_params": {},
  "compressor_params": {},
  "calculate_norm_of_gradients": True,
  "calculate_gradient_estimator_error": True,
  "calculate_function": True,
  "save_rate": 1000,
  "shared_memory_len": 2
}

def generate_yaml(dumps_path, dataset_path, experiments_name, dataset, num_nodes_list, compressor_to_step_size_range, number_of_seeds,
                  number_of_iterations, algorithm_names, func, parallel, cpus_per_task,
                  time, number_of_processes, compressors,
                  calculate_smoothness_variance,
                  scale_initial_point, reg_paramterer, ef21_init_with_gradients,
                  point_initializer, batch_size, homogeneous, number_of_coordinates,
                  quality_check_rate, logistic_regression_nonconvex,
                  gb_per_task, oracle, mega_batch_size, no_initial_mega_batch_size,
                  no_initial_mega_batch_size_zero, partial_participation_probabilities,
                  split_into_groups_by_labels, equalize_to_same_number_samples_per_class,
                  subsample_classes, fixed_seed):
    folder_with_project = os.getcwd()
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=dumps_path, experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=dumps_path, experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    script_to_execute = ''
    generator = np.random.default_rng(seed=42)
    print(compressor_to_step_size_range)
    index = 0
    partial_participation_probabilities_pairs = [[partial_participation_probability, False] 
                                           for partial_participation_probability in partial_participation_probabilities]
    partial_participation_probabilities_pairs[0][1] = True
    for _, (compressor_name, algorithm_name, num_nodes, partial_participation_probabilities_pair, seed_number) in enumerate(
        itertools.product(compressors, algorithm_names, num_nodes_list, partial_participation_probabilities_pairs, 
                          range(number_of_seeds))):
        step_size_range = compressor_to_step_size_range[compressor_name]
        for gamma_multiply in [2**i for i in range(step_size_range[0], step_size_range[1])]:
            yaml_prepared = copy.deepcopy(yaml_template)
            yaml_prepared['dataset_name'] = dataset
            if dataset == 'rcv1_test.binary_parsed':
                yaml_prepared['sparse_dataset'] = True
            yaml_prepared['num_nodes'] = num_nodes
            yaml_prepared['algorithm_name'] = algorithm_name
            yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
            yaml_prepared['compressor_name'] = compressor_name
            if compressor_name == 'rand_k':
                yaml_prepared['compressor_params']['number_of_coordinates'] = number_of_coordinates
            if dataset == 'real-sim' and func in 'stochastic_logistic_regression':
                dim = 2 * 20958
            elif dataset == 'mushrooms' and func in 'stochastic_logistic_regression':
                dim = 2 * 112
            elif dataset == 'mnist' and func in 'stochastic_logistic_regression':
                dim = 10 * 785
            elif dataset == 'aloi' and func in 'stochastic_logistic_regression':
                dim = 2 * 128
            elif dataset == 'real-sim' and func in 'nonconvex':
                dim = 20958
            elif dataset == 'mushrooms' and func in 'nonconvex':
                dim = 112
            elif dataset == 'aloi' and func in 'nonconvex':
                dim = 128
            else:
                raise RuntimeError()
            if oracle == 'minibatch':
                if algorithm_name != 'frecon_stochastic':
                    yaml_prepared['algorithm_master_params']['batch_size'] = batch_size
                else:
                    yaml_prepared['batch_size'] = batch_size
                if algorithm_name == 'frecon_stochastic':
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = batch_size
                if algorithm_name in ['zero_marina_partial_participation_page', 'frecon_stochastic',
                                      'vr_marina_partial_participation']:
                    partial_participation_probability = partial_participation_probabilities_pair[0]
                    yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
                elif not partial_participation_probabilities_pair[1]:
                    continue
            elif oracle == 'stochastic':
                if compressor_name == 'rand_k':
                    omega = dim / number_of_coordinates - 1
                elif compressor_name == 'identity_unbiased':
                    omega = np.float32(0.0)
                if algorithm_name in ['frecon_stochastic', 'zero_marina_partial_participation_stochastic',
                                      'marina_stochastic_partial_participation']:
                    partial_participation_probability = partial_participation_probabilities_pair[0]
                    yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
                elif not partial_participation_probabilities_pair[1]:
                    continue
                if algorithm_name == 'zero_marina_stochastic':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = min(1 / float(mega_batch_size), 1 / (omega * math.sqrt(mega_batch_size)))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega * math.sqrt(mega_batch_size)))
                elif algorithm_name == 'zero_marina_sync_stochastic':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = mega_batch_size
                    yaml_prepared['algorithm_master_params']['prob_sync'] = min(1 / float(mega_batch_size), 1 / (omega + 1))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega))
                elif algorithm_name == 'zero_marina_partial_participation_stochastic':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = partial_participation_probability * min(1 / float(mega_batch_size), 1 / (omega * math.sqrt(mega_batch_size)))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega * math.sqrt(mega_batch_size)) / math.sqrt(partial_participation_probability))
                elif algorithm_name == 'frecon_stochastic':
                    # The same as in dasha
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega * math.sqrt(mega_batch_size)) / math.sqrt(partial_participation_probability))
                elif algorithm_name == 'marina_stochastic':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = mega_batch_size
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(mega_batch_size, omega))
                elif algorithm_name == 'marina_stochastic_partial_participation':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = mega_batch_size
                else:
                    assert False
                if no_initial_mega_batch_size:
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = 1
                if no_initial_mega_batch_size_zero:
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = 0
                yaml_prepared['batch_size'] = batch_size
            elif oracle == 'gradient':
                if algorithm_name in ['zero_marina_partial_participation', 
                                      'frecon',
                                      'marina_partial_participation']:
                    partial_participation_probability = partial_participation_probabilities_pair[0]
                    yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
                # if algorithm_name == 'zero_marina_no_init':
                #     yaml_prepared['algorithm_master_params']['init_with_gradients'] = False
                #     yaml_prepared['algorithm_name'] = 'zero_marina'
                if algorithm_name == 'frecon':
                    yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
                if algorithm_name == 'marina_partial_participation':
                    yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
                # if algorithm_name == 'zero_marina_partial_participation_no_init':
                #     yaml_prepared['algorithm_master_params']['init_with_gradients'] = False
                #     yaml_prepared['algorithm_name'] = 'zero_marina_partial_participation'
                #     yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
                if algorithm_name == 'zero_marina_partial_participation':
                    yaml_prepared['algorithm_master_params']['number_of_samples'] = int(partial_participation_probability * num_nodes)
            yaml_prepared['number_of_iterations'] = number_of_iterations
            yaml_prepared['shuffle'] = True
            yaml_prepared['split_into_groups_by_labels'] = split_into_groups_by_labels
            yaml_prepared['equalize_to_same_number_samples_per_class'] = equalize_to_same_number_samples_per_class
            yaml_prepared['subsample_classes'] = subsample_classes
            yaml_prepared['ignore_remainder'] = True
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
            ibex_str = ibex_template.format(folder_with_project=folder_with_project,
                                            singularity_script=singularity_script_name,
                                            experiments_name=experiments_name,
                                            cpus_per_task=cpus_per_task,
                                            time=time,
                                            gb_per_task=gb_per_task)
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
    parser.add_argument('--dataset', required=True)
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
    parser.add_argument('--compressors', nargs='+', default=['rand_k'])
    parser.add_argument('--calculate_smoothness_variance', action='store_true')
    parser.add_argument('--point_initializer')
    parser.add_argument('--scale_initial_point', type=float)
    parser.add_argument('--reg_paramterer', type=float)
    parser.add_argument('--ef21_init_with_gradients', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--mega_batch_size', default=128, type=int)
    parser.add_argument('--number_of_coordinates', default=10, type=int)
    parser.add_argument('--homogeneous', action='store_true')
    parser.add_argument('--logistic_regression_nonconvex', default=0.0, type=float)
    parser.add_argument('--quality_check_rate', default=1, type=int)
    parser.add_argument('--oracle', default='gradient')
    parser.add_argument('--no_initial_mega_batch_size', action='store_true')
    parser.add_argument('--no_initial_mega_batch_size_zero', action='store_true')
    parser.add_argument('--partial_participation_probabilities', default=[1.0], type=float, nargs="+")
    parser.add_argument('--split_into_groups_by_labels', action='store_true')
    parser.add_argument('--equalize_to_same_number_samples_per_class', type=int)
    parser.add_argument('--subsample_classes', type=int)
    parser.add_argument('--fixed_seed', type=int)
    args = parser.parse_args()
    step_size_ranges = []
    for index in range(len(args.step_size_range) // 2):
        step_size_ranges.append([args.step_size_range[2 * index],
                                 args.step_size_range[2 * index + 1]])
    compressor_to_step_size_range = {}
    if len(step_size_ranges) == 1:
        for compressor in args.compressors:
            compressor_to_step_size_range[compressor] = step_size_ranges[0]
    else:
        assert len(step_size_ranges) == len(args.compressors)
        for compressor, step_size_range in zip(args.compressors, step_size_ranges):
            compressor_to_step_size_range[compressor] = step_size_range
    generate_yaml(args.dumps_path, args.dataset_path, args.experiments_name, args.dataset, args.num_nodes_list, compressor_to_step_size_range,
                  args.number_of_seeds, args.number_of_iterations, args.algorithm_names,
                  args.function, args.parallel, args.cpus_per_task, args.time,
                  args.number_of_processes, args.compressors,
                  args.calculate_smoothness_variance,
                  args.scale_initial_point, args.reg_paramterer,
                  args.ef21_init_with_gradients, args.point_initializer,
                  args.batch_size, args.homogeneous, args.number_of_coordinates,
                  args.quality_check_rate, args.logistic_regression_nonconvex,
                  args.gb_per_task, args.oracle, args.mega_batch_size,
                  args.no_initial_mega_batch_size, args.no_initial_mega_batch_size_zero,
                  args.partial_participation_probabilities,
                  args.split_into_groups_by_labels, args.equalize_to_same_number_samples_per_class,
                  args.subsample_classes, args.fixed_seed)
