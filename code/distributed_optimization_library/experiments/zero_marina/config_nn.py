import itertools
import tempfile
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
#SBATCH --cpus-per-task={cpus_per_task}

cd {folder_with_project}
module load singularity
singularity exec ~/rsync_watch/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif sh {singularity_script}
'''

singularity_template = 'CUDA_VISIBLE_DEVICES={device} PYTHONPATH=./code python3 code/distributed_optimization_library/experiments/experiments.py --path_to_dataset {dataset_path} --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "task": "cifar10",
  "algorithm_master_params": {},
  "calculate_function": True,
  "seed": 42,
  "statistics": True,
  "save_rate": 2000,
  "resnet_params": {"activation": None}
}

def generate_yaml(dumps_path, dataset_path, experiments_name, num_nodes_list, step_size_range,
                  number_of_iterations, algorithm_names, parallel, cpus_per_task,
                  time, number_of_processes, compressors,
                  batch_size, number_of_coordinates,
                  quality_check_rate, gb_per_task,
                  devices, noise_momentums,
                  resnet_activation, no_correction_noise_momentum,
                  mega_batch_sizes):
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=dumps_path, experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=dumps_path, experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    print("Tmp dir: {}".format(tmp_dir))
    script_to_execute = ''
    print(step_size_range)
    index = 0
    folder_with_project = os.getcwd()
    for _, (compressor_name, algorithm_name, num_nodes) in enumerate(
        itertools.product(compressors, algorithm_names, num_nodes_list)):
        if algorithm_name == 'zero_marina_stochastic':
            extra_parameters = noise_momentums
        elif algorithm_name == 'zero_marina_sync_stochastic':
            extra_parameters = mega_batch_sizes
        elif algorithm_name == 'marina_stochastic':
            extra_parameters = mega_batch_sizes
        dim = 10000000
        omega = dim / number_of_coordinates - 1
        for gamma in step_size_range:
            for extra_parameter in extra_parameters:
                yaml_prepared = copy.deepcopy(yaml_template)
                yaml_prepared['num_nodes'] = num_nodes
                yaml_prepared['resnet_params']['activation'] = resnet_activation
                yaml_prepared['algorithm_name'] = algorithm_name
                yaml_prepared['algorithm_master_params']['gamma'] = gamma
                if algorithm_name == 'zero_marina_stochastic':
                    yaml_prepared['algorithm_master_params']['noise_momentum'] = extra_parameter
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(int(1 / extra_parameter), omega**2))
                    yaml_prepared['compressor_name'] = compressor_name
                    yaml_prepared['compressor_params'] = {'is_cuda': True, 
                                                          'number_of_coordinates': number_of_coordinates}
                if algorithm_name == 'zero_marina_sync_stochastic':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = extra_parameter
                    yaml_prepared['algorithm_master_params']['prob_sync'] = min(1 / float(extra_parameter), 1 / (omega + 1))
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(extra_parameter, omega))
                    yaml_prepared['compressor_name'] = compressor_name
                    yaml_prepared['compressor_params'] = {'is_cuda': True, 
                                                          'number_of_coordinates': number_of_coordinates}
                if algorithm_name == 'marina_stochastic':
                    yaml_prepared['algorithm_master_params']['mega_batch_size'] = extra_parameter
                    yaml_prepared['algorithm_master_params']['initial_mega_batch_size'] = int(max(extra_parameter, omega))
                    yaml_prepared['compressor_name'] = compressor_name
                    yaml_prepared['compressor_params'] = {'is_cuda': True, 
                                                          'number_of_coordinates': number_of_coordinates}
                yaml_prepared['number_of_iterations'] = number_of_iterations
                yaml_prepared['shuffle'] = True
                yaml_prepared['shared_memory_size'] = 100000
                yaml_prepared['parallel'] = parallel
                yaml_prepared['multiprocessing_spawn'] = parallel
                yaml_prepared['number_of_processes'] = number_of_processes
                yaml_prepared['batch_size'] = batch_size
                yaml_prepared['quality_check_rate'] = quality_check_rate
                
                config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
                with open(config_name, 'w') as fd:
                    yaml.dump([yaml_prepared], fd)
                device = devices[index % len(devices)]
                singularity_script = singularity_template.format(experiments_name=experiments_name,
                                                                config_name=config_name,
                                                                device=device,
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
    parser.add_argument('--num_nodes_list', required=True, nargs='+', type=int)
    parser.add_argument('--step_size_range', required=True, nargs='+', type=float)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--algorithm_names', required=True, nargs='+')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--gb_per_task', default=30, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    parser.add_argument('--time', default=10, type=int)
    parser.add_argument('--compressors', nargs='+', default=['rand_k_torch'])
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--number_of_coordinates', default=10, type=int)
    parser.add_argument('--quality_check_rate', default=1, type=int)
    parser.add_argument('--devices', default=[0, 1, 2, 3], type=int, nargs='+')
    parser.add_argument('--noise_momentums', nargs='+', type=float, default=[None])
    parser.add_argument('--mega_batch_sizes', nargs='+', type=int, default=[None])
    parser.add_argument('--resnet_activation', default='elu')
    parser.add_argument('--no_correction_noise_momentum', action='store_true')
    args = parser.parse_args()
    generate_yaml(args.dumps_path, args.dataset_path, args.experiments_name, args.num_nodes_list, args.step_size_range,
                  args.number_of_iterations, args.algorithm_names,
                  args.parallel, args.cpus_per_task, args.time,
                  args.number_of_processes, args.compressors,
                  args.batch_size, args.number_of_coordinates,
                  args.quality_check_rate, args.gb_per_task,
                  args.devices, args.noise_momentums,
                  args.resnet_activation, args.no_correction_noise_momentum,
                  args.mega_batch_sizes)
