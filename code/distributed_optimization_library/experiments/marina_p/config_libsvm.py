import itertools
import tempfile
import os
import yaml
import copy
import numpy as np
import argparse
import sys

ibex_template = '''#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J MyJob{experiments_name}
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time={time}:30:00
#SBATCH --mem=30G
#SBATCH --cpus-per-task={cpus_per_task}

cd /home/tyurina/rsync_watch/distributed_optimization_library/
module load singularity
singularity exec ~/rsync_watch/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif sh {singularity_script}
'''

singularity_template = 'MKL_NUM_THREADS=1 PYTHONPATH=./code python3.7 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset ~/data --dumps_path ~/exepriments/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": "libsvm",
  "algorithm_master_params": {"gamma": 1.0},
  "compressor_params": {},
  "calculate_norm_of_gradients": True,
  "calculate_function": True,
  "calculate_gradient_estimator_error": False,
  "quality_check_rate": 20,
  "save_rate": 100,
}

def generate_yaml(experiments_name, dataset, num_nodes_list, compressor_to_step_size_range, number_of_seeds,
                  number_of_iterations, algorithm_name, func, parallel, cpus_per_task,
                  time, number_of_processes, algorithms, split_into_groups_by_labels,
                  calculate_smoothness_variance, split_with_controling_homogeneity,
                  scale_initial_point, reg_paramterer, split_with_all_dataset,
                  split_with_all_dataset_max_number, ef21_init_with_gradients,
                  point_initializer, core_number_of_coordinates, problem_type, square_norm_dim,
                  square_norm_variance):
    os.mkdir(os.path.abspath('/home/tyurina/exepriments/{experiments_name}'.format(experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('/home/tyurina/exepriments/{experiments_name}/source_folder'.format(experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    print("Tmp dir: {}".format(tmp_dir))
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    script_to_execute = ''
    generator = np.random.default_rng(seed=42)
    if square_norm_variance > 0:
        assert len(num_nodes_list) == 1
        random_values = generator.normal(size=(num_nodes_list[0],))
        random_values = (square_norm_variance * random_values).tolist()
    print(compressor_to_step_size_range)
    index = 0
    for _, (algorithm_name, num_nodes, seed_number) in enumerate(
        itertools.product(algorithms, num_nodes_list, range(number_of_seeds))):
        step_size_range = compressor_to_step_size_range[algorithm_name]
        for gamma_multiply in [2**i for i in range(step_size_range[0], step_size_range[1])]:
            yaml_prepared = copy.deepcopy(yaml_template)
            if problem_type == 'libsvm':
                yaml_prepared['dataset_name'] = dataset
            yaml_prepared['num_nodes'] = num_nodes
            yaml_prepared['algorithm_master_params']['gamma_multiply'] = gamma_multiply
            yaml_prepared['number_of_iterations'] = number_of_iterations
            yaml_prepared['shuffle'] = True
            yaml_prepared['homogeneous'] = False
            yaml_prepared['shared_memory_size'] = 100000
            yaml_prepared['parallel'] = parallel
            yaml_prepared['number_of_processes'] = number_of_processes
            yaml_prepared['function'] = func
            yaml_prepared['seed'] = seed_number
            if problem_type == 'libsvm':
                yaml_prepared['split_into_groups_by_labels'] = split_into_groups_by_labels
                yaml_prepared['split_with_controling_homogeneity'] = None
                yaml_prepared['split_with_all_dataset'] = None
                if split_with_controling_homogeneity is not None and split_with_controling_homogeneity > 0.0:
                    yaml_prepared['split_with_controling_homogeneity'] = split_with_controling_homogeneity
                if split_with_all_dataset is not None:
                    assert split_with_all_dataset > 0.0
                    yaml_prepared['split_with_all_dataset'] = split_with_all_dataset
                    yaml_prepared['split_with_all_dataset_max_number'] = split_with_all_dataset_max_number
                if scale_initial_point is not None:
                    assert scale_initial_point > 0.0
                    yaml_prepared['scale_initial_point'] = scale_initial_point
                if point_initializer is not None:
                    yaml_prepared['point_initializer'] = point_initializer
                if reg_paramterer is not None and reg_paramterer > 0.0:
                    yaml_prepared['reg_paramterer'] = reg_paramterer
                
                dim = None
                if yaml_prepared['function'] in ['nonconvex_multiclass']:
                    if yaml_prepared['dataset_name'] == 'gisette_scale':
                        dim = 5000
                    elif yaml_prepared['dataset_name'] == 'mnist':
                        dim = 7850
                if yaml_prepared['function'] in ['two_layer_neural_net',
                                                'two_layer_neural_net_relu']:
                    if yaml_prepared['dataset_name'] == 'mnist':
                        dim = 25450
                    if yaml_prepared['dataset_name'] == 'mushrooms':
                        dim = 3682
                if yaml_prepared['function'] in ['two_layer_neural_net_skip_connection']:
                    if yaml_prepared['dataset_name'] == 'mnist':
                        dim = 26506
                if yaml_prepared['function'] in ['two_layer_neural_net_linear']:
                    if yaml_prepared['dataset_name'] == 'mnist':
                        dim = 7850
                    if yaml_prepared['dataset_name'] == 'mushrooms':
                        dim = 226
                if yaml_prepared['function'] in ['two_layer_neural_net_worst_case']:
                    if yaml_prepared['dataset_name'] == 'mushrooms':
                        dim = 1554
                if yaml_prepared['function'] in ['two_layer_neural_net_worst_case_sigmoid']:
                    if yaml_prepared['dataset_name'] == 'mushrooms':
                        dim = 1346
                if yaml_prepared['function'] in ['auto_encoder']:
                    if yaml_prepared['dataset_name'] == 'mnist':
                        dim = 25088
                if yaml_prepared['function'] in ['auto_encoder_equal']:
                    if yaml_prepared['dataset_name'] == 'mnist':
                        dim = 12544
                if yaml_prepared['function'] in ['logistic_regression']:
                    if yaml_prepared['dataset_name'] == 'mnist':
                        dim = 7850
                    if yaml_prepared['dataset_name'] == 'real-sim':
                        dim = 41918
                assert dim is not None
            
                yaml_prepared['dim'] = dim
            elif problem_type == 'square_norm':
                yaml_prepared['dim'] = square_norm_dim
                yaml_prepared['task'] = 'quadratic'
                if square_norm_variance > 0:
                    yaml_prepared['square_norm_lipt'] = random_values
                dim = square_norm_dim
            else:
                assert False
            
            if algorithm_name == 'm3_perm_rand':
                yaml_prepared['compressor_name'] = 'rand_k'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = 'permutation'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
            elif algorithm_name == 'm3_perm_with_nat_rand_with_nat':
                yaml_prepared['compressor_name'] = '_rand_k_with_nat'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
            elif algorithm_name == 'm3_perm_with_nat_rand_with_nat_comp':
                yaml_prepared['compressor_name'] = '_rand_k_with_nat'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
                yaml_prepared["algorithm_master_params"]["use_vector_compression"] = True
            elif algorithm_name == 'm3_perm_with_nat_rand_with_nat_ps_0_5':
                yaml_prepared['compressor_name'] = '_rand_k_with_nat'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
                yaml_prepared["algorithm_master_params"]["prob_scale"] = 0.5
            elif algorithm_name == 'm3_perm_with_nat_rand_with_nat_psp_0_5':
                yaml_prepared['compressor_name'] = '_rand_k_with_nat'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
                yaml_prepared["algorithm_master_params"]["prob_scale_primal"] = 0.5
            elif algorithm_name == 'm3_perm_with_nat_rand_with_nat_psp_0_5_ps_0_5':
                yaml_prepared['compressor_name'] = '_rand_k_with_nat'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
                yaml_prepared["algorithm_master_params"]["prob_scale"] = 0.5
                yaml_prepared["algorithm_master_params"]["prob_scale_primal"] = 0.5
            # elif algorithm_name == 'm3_perm_rand_prob_4':
            #     yaml_prepared['compressor_name'] = 'rand_k'
            #     yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
            #     yaml_prepared['compressor_master_name'] = 'permutation'
            #     yaml_prepared['compressor_master_params'] = {}
            #     yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
            #     yaml_prepared['multiple_master_compressors'] = True
            #     yaml_prepared["algorithm_name"] = 'm3'
            #     yaml_prepared["algorithm_master_params"]["prob_scale"] = 0.25
            # elif algorithm_name == 'm3_perm_rand_prob_m_2':
            #     yaml_prepared['compressor_name'] = 'rand_k'
            #     yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
            #     yaml_prepared['compressor_master_name'] = 'permutation'
            #     yaml_prepared['compressor_master_params'] = {}
            #     yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
            #     yaml_prepared['multiple_master_compressors'] = True
            #     yaml_prepared["algorithm_name"] = 'm3'
            #     yaml_prepared["algorithm_master_params"]["prob_scale"] = 2
            elif algorithm_name == 'm3_perm_core':
                yaml_prepared['compressor_name'] = 'core'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = 'permutation'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
            elif algorithm_name == 'm3_perm_perm':
                yaml_prepared['compressor_name'] = 'permutation'
                yaml_prepared['compressor_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['compressor_master_name'] = 'permutation'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
            elif algorithm_name == 'm3_ind_rand': # it should be m3_perm_ind
                yaml_prepared['compressor_name'] = 'identity_unbiased'
                yaml_prepared['compressor_master_name'] = 'permutation'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
            elif algorithm_name == 'm3_perm_nat_ind':
                yaml_prepared['compressor_name'] = 'identity_unbiased'
                yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
                yaml_prepared['multiple_master_compressors'] = True
                yaml_prepared["algorithm_name"] = 'm3'
            # elif algorithm_name == 'm3_perm_nat_ind_prob_2':
            #     yaml_prepared['compressor_name'] = 'identity_unbiased'
            #     yaml_prepared['compressor_master_name'] = '_permutation_with_nat'
            #     yaml_prepared['compressor_master_params'] = {}
            #     yaml_prepared['compressor_master_params']['total_number_of_nodes'] = yaml_prepared['num_nodes']
            #     yaml_prepared['multiple_master_compressors'] = True
            #     yaml_prepared["algorithm_name"] = 'm3'
            #     yaml_prepared["algorithm_master_params"]["prob_scale"] = 0.5
            elif algorithm_name == 'dcgd_ef21_primal_rand_rand':
                yaml_prepared['compressor_name'] = 'rand_k'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = 'biased_rand_k'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared["algorithm_name"] = 'dcgd_ef21_primal'
            elif algorithm_name == 'dcgd_ef21_primal_rand_nat_rand_nat':
                yaml_prepared['compressor_name'] = '_rand_k_with_nat'
                yaml_prepared['compressor_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared['compressor_master_name'] = '_biased_rand_k_with_nat'
                yaml_prepared['compressor_master_params'] = {}
                yaml_prepared['compressor_master_params']['number_of_coordinates'] = int(dim / yaml_prepared['num_nodes'])
                yaml_prepared["algorithm_name"] = 'dcgd_ef21_primal'
            elif algorithm_name == 'gradient_descent':
                yaml_prepared["algorithm_name"] = 'gradient_descent'
            elif algorithm_name == 'core':
                yaml_prepared["algorithm_name"] = 'core'
                yaml_prepared["algorithm_node_params"] = {'number_of_coordinates': core_number_of_coordinates}
            else:
                raise RuntimeError()
            
            config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
            with open(config_name, 'w') as fd:
                yaml.dump([yaml_prepared], fd)
            singularity_script = singularity_template.format(experiments_name=experiments_name,
                                                            config_name=config_name)
            singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
            open(singularity_script_name, 'w').write(singularity_script)
            ibex_str = ibex_template.format(singularity_script=singularity_script_name,
                                            experiments_name=experiments_name,
                                            cpus_per_task=cpus_per_task,
                                            time=time)
            script_name = os.path.join(tmp_dir, 'script_{}.sh'.format(index))
            open(script_name, 'w').write(ibex_str)
            script_to_execute += 'sbatch {}\n'.format(script_name)
            index += 1
    final_script_name = os.path.join(tmp_dir, 'execute.sh')
    open(final_script_name, 'w').write(script_to_execute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--num_nodes_list', required=True, nargs='+', type=int)
    parser.add_argument('--step_size_range', required=True, nargs='+', type=int)
    parser.add_argument('--number_of_seeds', type=int, default=1)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--algorithm_name', default="rand_diana")
    parser.add_argument('--function', default="logistic_regression")
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    parser.add_argument('--time', default=10, type=int)
    parser.add_argument('--algorithms', nargs='+', default=['m3'])
    parser.add_argument('--split_into_groups_by_labels', action='store_true')
    parser.add_argument('--split_with_controling_homogeneity', type=float)
    parser.add_argument('--split_with_all_dataset_max_number', type=int, default=1000)
    parser.add_argument('--split_with_all_dataset', type=float)
    parser.add_argument('--calculate_smoothness_variance', action='store_true')
    parser.add_argument('--point_initializer')
    parser.add_argument('--scale_initial_point', type=float)
    parser.add_argument('--reg_paramterer', type=float)
    parser.add_argument('--ef21_init_with_gradients', action='store_true')
    parser.add_argument('--core_number_of_coordinates', type=int, default=100)
    parser.add_argument('--problem_type', default='libsvm')
    parser.add_argument('--square_norm_dim', type=int, default=1)
    parser.add_argument('--square_norm_variance', type=float, default=0)
    args = parser.parse_args()
    step_size_ranges = []
    for index in range(len(args.step_size_range) // 2):
        step_size_ranges.append([args.step_size_range[2 * index],
                                 args.step_size_range[2 * index + 1]])
    compressor_to_step_size_range = {}
    if len(step_size_ranges) == 1:
        for compressor in args.algorithms:
            compressor_to_step_size_range[compressor] = step_size_ranges[0]
    else:
        assert len(step_size_ranges) == len(args.algorithms)
        for compressor, step_size_range in zip(args.algorithms, step_size_ranges):
            compressor_to_step_size_range[compressor] = step_size_range
    generate_yaml(args.experiments_name, args.dataset, args.num_nodes_list, compressor_to_step_size_range,
                  args.number_of_seeds, args.number_of_iterations, args.algorithm_name,
                  args.function, args.parallel, args.cpus_per_task, args.time,
                  args.number_of_processes, args.algorithms, args.split_into_groups_by_labels,
                  args.calculate_smoothness_variance, args.split_with_controling_homogeneity,
                  args.scale_initial_point, args.reg_paramterer,
                  args.split_with_all_dataset, args.split_with_all_dataset_max_number,
                  args.ef21_init_with_gradients, args.point_initializer,
                  args.core_number_of_coordinates, args.problem_type,
                  args.square_norm_dim, args.square_norm_variance)
