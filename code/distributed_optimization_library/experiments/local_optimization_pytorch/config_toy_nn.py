import itertools
import tempfile
import os
import sys
import yaml
import copy
import numpy as np
import argparse

singularity_template = 'PYTHONPATH=./code python3.7 code/distributed_optimization_library/experiments/local_optimization_pytorch/optimize_toy_model.py --dumps_path {dumps_path}/{experiments_name}/ --config {config_name}'

yaml_template = {
  "seed": 42,
}

def generate_yaml(args):
    os.mkdir(os.path.abspath('{dumps_path}/{experiments_name}'.format(dumps_path=args.dumps_path, experiments_name=args.experiments_name)))
    tmp_dir = os.path.abspath('{dumps_path}/{experiments_name}/source_folder'.format(dumps_path=args.dumps_path, experiments_name=args.experiments_name))
    os.mkdir(tmp_dir)
    open(os.path.join(tmp_dir, 'cmd.txt'), 'w').write(" ".join(sys.argv))
    print("Tmp dir: {}".format(tmp_dir))
    script_to_execute = ''
    print(args.step_sizes)
    index = 0
    for extra_parameter in args.extra_parameters:
        for gamma in args.step_sizes:
            yaml_prepared = copy.deepcopy(yaml_template)
            yaml_prepared['seed'] = args.seed
            yaml_prepared['number_of_batches'] = args.number_of_iterations
            yaml_prepared['learning_rate'] = gamma
            yaml_prepared['batch_size'] = args.batch_size
            yaml_prepared['optimizer'] = args.optimizer
            yaml_prepared['save_every'] = args.save_every
            yaml_prepared['save_model_every'] = args.save_model_every
            yaml_prepared['batch_size_save_every'] = args.batch_size_save_every
            yaml_prepared['model_type'] = args.model_type
            if args.model_type in ['symmetric_logistic_loss', 'sqrt_one_plus', 'square_one_minus_square']:
                yaml_prepared['model_params'] = {}
                yaml_prepared['model_params']['starting_point'] = args.symmetric_logistic_loss_starting_point
            config_name = os.path.join(tmp_dir, 'config_{}.yaml'.format(index))
            with open(config_name, 'w') as fd:
                yaml.dump([yaml_prepared], fd)
            singularity_script = singularity_template.format(experiments_name=args.experiments_name,
                                                            config_name=config_name,
                                                            dumps_path=args.dumps_path)
            singularity_script_name = os.path.join(tmp_dir, 'singularity_{}.sh'.format(index))
            open(singularity_script_name, 'w').write(singularity_script)
            log_name = os.path.join(tmp_dir, 'log.txt'.format(index))
            script_to_execute += 'rm -f {} \n exec &>> {} \n sh {} &\n'.format(log_name, log_name, singularity_script_name)
            index += 1
        final_script_name = os.path.join(tmp_dir, 'execute.sh')
        open(final_script_name, 'w').write(script_to_execute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_path', required=True)
    parser.add_argument('--experiments_name', required=True)
    parser.add_argument('--step_sizes', required=True, nargs='+', type=float)
    parser.add_argument('--number_of_iterations', type=int, default=20000)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--extra_parameters', default=[None], nargs='+', type=float)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--save_every', default=100, type=int)
    parser.add_argument('--batch_size_save_every', default=128, type=int)
    parser.add_argument('--save_model_every', default=None, type=int)
    parser.add_argument('--model_type', default='linear')
    parser.add_argument('--seed', default=42)
    parser.add_argument('--symmetric_logistic_loss_starting_point', default=[0, 0], nargs='+', type=float)
    args = parser.parse_args()
    generate_yaml(args)
