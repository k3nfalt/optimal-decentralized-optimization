import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from collections import namedtuple

from distributed_optimization_library.experiments.local_optimization_pytorch.optimize_model import \
    prepare_dataset, StochasticModel
from distributed_optimization_library.models.small_models_eos import prepare_two_layer_nn_eos
from distributed_optimization_library.experiments.local_optimization_pytorch.optimize_model import \
    GradientDescent, AdaptiveGradientDescent

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def main(args):
    number_of_iterations = args.number_of_iterations
    path_to_dataset = "/home/tyurina/data"

    for number_of_first_examples in args.number_of_samples:

        trainset = prepare_dataset(path_to_dataset, number_of_first_examples=number_of_first_examples)
        print(np.unique(np.array(trainset.dataset.targets)[np.array(trainset.indices)], return_counts=True))
        optimize_memory = False

        def prepare_model_wrapper():
            loss_fn = torch.nn.CrossEntropyLoss()
            model = prepare_two_layer_nn_eos(activation='elu', type=args.model_name)
            model_wrapper = StochasticModel(model, loss_fn, trainset, batch_size=1,
                                            num_workers=4, use_cuda=True, optimize_memory=optimize_memory)
            return model_wrapper
            
        OptimizerParams = namedtuple("Point", "optimizer params")
        name_to_class = {'GD': GradientDescent, 'Adapt GD': AdaptiveGradientDescent}
        optimizer_params = [OptimizerParams('GD', {'lr': lr}) for lr in args.gd_lr]
        optimizer_params += [OptimizerParams('Adapt GD', {'lr': 100.0, 'stable': True})]
        model_wrappers = [prepare_model_wrapper() for _ in range(len(optimizer_params))]
        starting_point = model_wrappers[0].current_point()

        fig, ax = plt.subplots(figsize=(40, 20))

        for optimizer_param, model_wrapper in zip(optimizer_params, model_wrappers):
            optimizer_wrapper = name_to_class[optimizer_param.optimizer](
                model_wrapper, starting_point, **optimizer_param.params)
            losses = []
            accuracies = []
            for index in range(number_of_iterations):
                optimizer_wrapper.step()
                loss, accuracy = model_wrapper.last_loss_and_accuracy()
                losses.append(losses)
                accuracies.append(accuracy)
            ax.plot(range(len(accuracies)), accuracies, 
                    label="Optimizer: {} LR: {}".format(optimizer_param.optimizer, optimizer_param.params['lr']), 
                    linewidth=6)

        ax.legend(fontsize=30, loc='upper right')
        ax.yaxis.set_tick_params(labelsize=40)
        ax.xaxis.set_tick_params(labelsize=40)
        fig.savefig(os.path.join(args.path_to_save, "{}.pdf".format(number_of_first_examples)), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_save', required=True)
    parser.add_argument('--number_of_samples', nargs='*', type=int)
    parser.add_argument('--model_name', default='conv_stride_2_5_layers')
    parser.add_argument('--gd_lr', nargs='*', type=float)
    parser.add_argument('--number_of_iterations', type=int, default=5000)
    args = parser.parse_args()
    os.mkdir(args.path_to_save)
    with open(os.path.join(args.path_to_save, "cmd.txt"), 'w') as fd:
        fd.write(" ".join(sys.argv[:]))
    main(args)
