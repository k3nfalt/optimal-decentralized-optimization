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
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu

cd {folder_with_project}
module load singularity
singularity exec --nv ~/rsync_watch/python_ml_sha256.37fc4c8d86b92f0ac80f7a3a729d2a3d0294ea3c3895957bc7f647f1ef922745.sif sh {singularity_script}
'''

singularity_template = 'MKL_NUM_THREADS=1 {device} PYTHONPATH=./code python3.7 code/distributed_optimization_library/experiments/local_optimization_pytorch/optimize_model.py --path_to_dataset {dataset_path} --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "seed": 42,
}

def generate_yaml(dumps_path, dataset_path, dataset, experiments_name, step_sizes,
                  number_of_iterations,
                  parallel, cpus_per_task, time,
                  number_of_processes,
                  batch_size,
                  gb_per_task, devices,
                  extra_parameters, optimizer,
                  save_every, estimate_sigma, model,
                  mean_and_sigma, batch_size_save_every,
                  resnet_activation, no_first_conv, no_first_activation, no_avg_pool,
                  blocks, num_workers, init_gradient_estimator,
                  use_cuda, use_double, save_model_every,
                  first_5000_examples, holder_constant, local_run, num_features,
                  not_optimize_memory, eos_type, number_of_samples, loss_function,
                  binary_classification, filter_classes, seed, adaptive_stable_gd_delta):
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=dumps_path, experiments_name=experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=dumps_path, experiments_name=experiments_name))
    os.mkdir(tmp_dir)
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    print("Tmp dir: {}".format(tmp_dir))
    script_to_execute = ''
    print(step_sizes)
    index = 0
    folder_with_project = os.getcwd()
    # for _, (compressor_name, algorithm_name) in enumerate(
    #     itertools.product(compressors, algorithm_names)):
    for extra_parameter in extra_parameters:
        for gamma in step_sizes:
            yaml_prepared = copy.deepcopy(yaml_template)
            yaml_prepared['seed'] = seed
            yaml_prepared['num_workers'] = num_workers
            yaml_prepared['dataset'] = dataset
            yaml_prepared['learning_rate'] = gamma
            yaml_prepared['number_of_batches'] = number_of_iterations
            yaml_prepared['batch_size'] = batch_size
            yaml_prepared['optimizer'] = optimizer
            yaml_prepared['save_every'] = save_every
            yaml_prepared['save_model_every'] = save_model_every
            yaml_prepared['estimate_sigma'] = estimate_sigma
            yaml_prepared['mean_and_sigma'] = mean_and_sigma
            yaml_prepared['model'] = model
            yaml_prepared['use_cuda'] = use_cuda
            yaml_prepared['use_double'] = use_double
            yaml_prepared['batch_size_save_every'] = batch_size_save_every
            yaml_prepared['first_5000_examples'] = first_5000_examples
            yaml_prepared['binary_classification'] = binary_classification
            yaml_prepared['filter_classes'] = filter_classes
            yaml_prepared['number_of_samples'] = number_of_samples
            yaml_prepared['optimize_memory'] = not not_optimize_memory
            yaml_prepared['loss_function'] = loss_function
            if model in ['resnet_without_bn']:
                yaml_prepared["resnet_params"] = {}
                yaml_prepared['resnet_params']['activation'] = resnet_activation
                yaml_prepared['resnet_params']['first_conv'] = not no_first_conv
                yaml_prepared['resnet_params']['first_activation'] = not no_first_activation
                yaml_prepared['resnet_params']['avg_pool'] = not no_avg_pool
                yaml_prepared['resnet_params']['blocks'] = blocks
                yaml_prepared['resnet_params']['num_features'] = num_features
            if model in ['resnet']:
                yaml_prepared["resnet_params"] = {}
                yaml_prepared['resnet_params']['activation'] = resnet_activation
            if model in ['two_layer_nn']:
                yaml_prepared["two_layer_nn_params"] = {}
                yaml_prepared['two_layer_nn_params']['activation'] = resnet_activation
            if model in ['two_layer_nn_eos']:
                yaml_prepared["two_layer_nn_params_eos"] = {}
                yaml_prepared['two_layer_nn_params_eos']['activation'] = resnet_activation
                yaml_prepared['two_layer_nn_params_eos']['type'] = eos_type
            if model in ['two_layer_nn_paper']:
                yaml_prepared["two_layer_nn_params_paper"] = {}
                yaml_prepared['two_layer_nn_params_paper']['activation'] = resnet_activation
                yaml_prepared['two_layer_nn_params_paper']['type'] = eos_type
            if optimizer == 'sgd':
                yaml_prepared['momentum'] = extra_parameter
            elif optimizer == 'mvr':
                yaml_prepared['momentum'] = extra_parameter
                yaml_prepared['init_gradient_estimator'] = init_gradient_estimator
            elif optimizer in ['gd', 'debug_gd', 'gd_tmp_3000']:
                assert len(extra_parameters) == 1
            elif optimizer == 'sign_gd':
                assert len(extra_parameters) == 1
            elif optimizer == 'adaptive_gd' or optimizer == 'adaptive_stable_gd':
                yaml_prepared['adaptive_stable_gd_delta'] = adaptive_stable_gd_delta
                assert len(extra_parameters) == 1
            elif optimizer == 'holder_gd':
                yaml_prepared['holder_constant'] = holder_constant
            elif optimizer == 'adaptive_holder_gd':
                yaml_prepared['holder_constant'] = holder_constant
            else:
                raise RuntimeError()
            
            config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
            with open(config_name, 'w') as fd:
                yaml.dump([yaml_prepared], fd)
            if local_run:
                device = devices[index % len(devices)]
                device = "CUDA_VISIBLE_DEVICES={}".format(device)
            else:
                device = ""
            singularity_script = singularity_template.format(experiments_name=experiments_name,
                                                            config_name=config_name,
                                                            device=device,
                                                            dumps_path=dumps_path,
                                                            dataset_path=dataset_path)
            singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
            open(singularity_script_name, 'w').write(singularity_script)
            if not local_run:
                ibex_str = ibex_template.format(folder_with_project=folder_with_project,
                                                singularity_script=singularity_script_name,
                                                experiments_name=experiments_name,
                                                cpus_per_task=cpus_per_task,
                                                time=time,
                                                gb_per_task=gb_per_task)
                script_name = os.path.join(tmp_dir, 'script_{}.sh'.format(index))
                open(script_name, 'w').write(ibex_str)
                script_to_execute += 'sbatch {}\n'.format(script_name)
            else:
                log_name = os.path.join(tmp_dir, 'log.txt'.format(index))
                script_to_execute += 'rm -f {} \n exec &>> {} \n sh {} &\n'.format(log_name, log_name, singularity_script_name)
            index += 1
        final_script_name = os.path.join(tmp_dir, 'execute.sh')
        open(final_script_name, 'w').write(script_to_execute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--step_sizes', required=True, nargs='+', type=float)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--cpus_per_task', default=1, type=int)
    parser.add_argument('--gb_per_task', default=30, type=int)
    parser.add_argument('--number_of_processes', default=1, type=int)
    parser.add_argument('--time', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--devices', default=[0, 1, 2, 3], type=int, nargs='+')
    parser.add_argument('--extra_parameters', default=[None], nargs='+', type=float)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--model', default='resnet')
    parser.add_argument('--resnet_activation', default='relu')
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--batch_size_save_every', default=128, type=int)
    parser.add_argument('--estimate_sigma', action='store_true')
    parser.add_argument('--mean_and_sigma', action='store_true')
    parser.add_argument('--no_first_conv', action='store_true')
    parser.add_argument('--no_first_activation', action='store_true')
    parser.add_argument('--no_avg_pool', action='store_true')
    parser.add_argument('--blocks', default=[2, 2, 2, 2], type=int, nargs='+')
    parser.add_argument('--num_features', default=64, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--init_gradient_estimator', action='store_true')
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_double', action='store_true')
    parser.add_argument('--save_model_every', default=None, type=int)
    parser.add_argument('--first_5000_examples', action='store_true')
    parser.add_argument('--binary_classification', action='store_true')
    parser.add_argument('--number_of_samples', type=int, default=None)
    parser.add_argument('--filter_classes', type=int, nargs='*', default=None)
    parser.add_argument('--holder_constant', type=float, default=1.0)
    parser.add_argument('--adaptive_stable_gd_delta', type=float, default=1e-8)
    parser.add_argument('--local_run', action='store_true')
    parser.add_argument('--not_optimize_memory', action='store_true')
    parser.add_argument('--eos_type', default='linear')
    parser.add_argument('--loss_function', default='cross_entropy')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    generate_yaml(args.dumps_path, args.dataset_path, args.dataset, args.experiments_name, args.step_sizes,
                  args.number_of_iterations,
                  args.parallel, args.cpus_per_task, args.time,
                  args.number_of_processes,
                  args.batch_size,
                  args.gb_per_task, args.devices,
                  args.extra_parameters, args.optimizer,
                  args.save_every, args.estimate_sigma,
                  args.model, args.mean_and_sigma,
                  args.batch_size_save_every,
                  args.resnet_activation,
                  args.no_first_conv, args.no_first_activation, args.no_avg_pool,
                  args.blocks, args.num_workers, args.init_gradient_estimator,
                  args.use_cuda, args.use_double, args.save_model_every,
                  args.first_5000_examples, args.holder_constant, args.local_run,
                  args.num_features, args.not_optimize_memory, args.eos_type,
                  args.number_of_samples, args.loss_function, args.binary_classification,
                  args.filter_classes, args.seed, args.adaptive_stable_gd_delta)
