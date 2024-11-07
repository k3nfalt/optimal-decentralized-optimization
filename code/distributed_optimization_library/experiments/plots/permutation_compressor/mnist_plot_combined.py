import argparse
import os
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import decimal


def get_split_with_controling_homogeneity(dump):
    split_with_controling_homogeneity = dump['params']['config'].get('split_with_controling_homogeneity', 0.0)
    if split_with_controling_homogeneity is None:
        return 0.0
    return split_with_controling_homogeneity


# create a new context for this task
ctx = decimal.Context()

# 20 digits should be enough for everyone :D
ctx.prec = 20

def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def plot_results(output_path, zero_lambda):
    # dumps_paths = ['/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_001-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_001-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_001-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_001-ef21-init-grad-xavier-normal/',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_00001-ef21-init-grad-xavier-normal-repeat',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_00001-ef21-init-grad-xavier-normal-repeat',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_00001-ef21-init-grad-xavier-normal-repeat',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_00001-ef21-init-grad-xavier-normal-repeat']
    
    # dumps_paths = ['/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_00001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_0-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_00001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_0-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_00001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_0-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_00001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_00001-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk']
    
    dumps_paths = ['/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
                   '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm']
    
    # dumps_paths = ['/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm']
    
    # dumps_paths = ['/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_00001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk-fix-perm',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_0-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_00001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_0-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_00001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_5-reg-0_001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_0-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_00001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_75-reg-0_001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_0-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_00001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_9-reg-0_001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_0-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_00001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_001-ef21-init-grad-xavier-normal-perm-fixed-blocks',
    #                ]
    
    # dumps_paths = ['/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-0_0-reg-0_0-ef21-init-grad-xavier-normal-with-randk',
    #                '/home/tyurina/exepriments/distributed_optimization_library-marina-mnist-auto_encoder-1000-nodes-prob-homog-1_0-reg-0_001-ef21-init-grad-xavier-normal-with-randk']
    
    dumps = []
    for dumps_path in dumps_paths:
        files = os.listdir(dumps_path)
        files = [(x, dumps_path) for x in files]
        dumps += files
    
    unique_reg_paramterer = set()
    unique_split_with_controling_homogeneity = set()
    dumps_loaded = []
    for dump, dumps_path in dumps:
        if dump == "source_folder":
            continue
        dump = json.load(open(os.path.join(dumps_path, dump)))
        if zero_lambda and dump['params']['config'].get('reg_paramterer', 0.0) != 0.0:
            continue
        unique_reg_paramterer.add(dump['params']['config'].get('reg_paramterer', 0.0))
        unique_split_with_controling_homogeneity.add(get_split_with_controling_homogeneity(dump))
        dumps_loaded.append(dump)
    unique_reg_paramterer = sorted(list(unique_reg_paramterer))
    unique_split_with_controling_homogeneity = sorted(list(unique_split_with_controling_homogeneity), reverse=True)
    print(unique_reg_paramterer, unique_split_with_controling_homogeneity)
    
    if not zero_lambda:
        fig, axs = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(42, 22))
        fig_2, axs_2 = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(42, 22))
        fig_best, axs_best = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(42, 22))
    else:
        # fig, axs = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(75, 13))
        # fig_2, axs_2 = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(75, 13))
        # fig_best, axs_best = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(75, 13))
        fig, axs = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(70, 7))
        fig_2, axs_2 = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(70, 7))
        fig_best, axs_best = plt.subplots(len(unique_reg_paramterer), len(unique_split_with_controling_homogeneity), figsize=(70, 7))
    # if str(type(axs[0])) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
    #     axs = [axs]
    #     axs_best = [axs_best]
    # axs = [axs]
    # axs_2 = [axs_2]
    # axs_best = [axs_best]
    if zero_lambda:
        axs = [axs]
        axs_2 = [axs_2]
        axs_best = [axs_best]
        # axs = [[x] for x in axs]
        # axs_2 = [[x] for x in axs_2]
        # axs_best = [[x] for x in axs_best]
    best_experiment = [[{} for _ in range(len(unique_split_with_controling_homogeneity))] for _ in range(len(unique_reg_paramterer))]
    ignored = [[{} for _ in range(len(unique_split_with_controling_homogeneity))] for _ in range(len(unique_reg_paramterer))]

    representation_name = {'rand_k': 'RandK',
                           'permutation': 'PermK',
                           'permutation_fixed_blocks': 'PermK (Fixed Blocks)',
                           'top_k': 'TopK'}
    
    index = 0
    for dump in dumps_loaded:
        dump['_index'] = index
        i = unique_reg_paramterer.index(dump['params']['config'].get('reg_paramterer', 0.0))
        j = unique_split_with_controling_homogeneity.index(get_split_with_controling_homogeneity(dump))
        dd = len(dump['stat']['function_values'])
        min_function_values = np.nanmean(np.log10(dump['stat']['function_values'])[int(0.1 * dd) : ])
        compressor_name = dump['params']['config']['compressor_name']
        if compressor_name not in best_experiment[i][j]:
            best_experiment[i][j][compressor_name] = []
            ignored[i][j][compressor_name] = []
        if not np.isnan(dump['stat']['function_values']).any() and not np.isinf(dump['stat']['function_values']).any():
            best_experiment[i][j][compressor_name].append((min_function_values, dump['_index']))
        else:
            ignored[i][j][compressor_name].append(dump)
        index += 1
    # for i in range(len(unique_reg_paramterer)):
    #     for j in range(len(unique_split_with_controling_homogeneity)):
    #         print("!!!!" * 100)
    #         print(unique_reg_paramterer[i], unique_split_with_controling_homogeneity[j])
    #         for compressor_name in ignored[i][j]:
    #             min_ = float('inf')
    #             for dump in ignored[i][j][compressor_name]:
    #                 min_ = min(min_, dump['params']['config']['algorithm_master_params']['gamma_multiply'])
    #                 # print(dump['params']['config'].get('reg_paramterer', 0.0))
    #                 # print(get_split_with_controling_homogeneity(dump))
    #                 # print(dump['params']['config']['compressor_name'])
    #                 # print(dump['params']['config']['algorithm_master_params']['gamma_multiply'])
    #             print(compressor_name, min_)
    top_show = 3
    use_experiment = [[defaultdict(set) 
                       for _ in range(len(unique_split_with_controling_homogeneity))] 
                      for _ in range(len(unique_reg_paramterer))]
    for i in range(len(best_experiment)):
        for j in range(len(best_experiment[0])):
            for comp in best_experiment[i][j]:
                best_experiment[i][j][comp] = sorted(best_experiment[i][j][comp])
                for k in range(min(len(best_experiment[i][j][comp]), top_show)):
                    use_experiment[i][j][comp].add(best_experiment[i][j][comp][k][1])
    dumps_loaded = sorted(dumps_loaded, 
                          key=lambda dump: (dump['params']['config']['compressor_name'],
                                            dump['params']['config']['algorithm_master_params']['gamma_multiply']))
    
    log_index = [[0] *len(unique_split_with_controling_homogeneity) for _ in range(len(unique_reg_paramterer))]
    processed_index = [[0] *len(unique_split_with_controling_homogeneity) for _ in range(len(unique_reg_paramterer))]
    markers = ['v','^','<','>','s','p','P','*','h','H','+','x','X','D','d','|','_']
    markers += markers
    markers += markers
    colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors += colors
    colors += colors
    best_experiments_to_markers = {'rand_k': {'marker': '<', 'color': 'red'},
                                   'permutation': {'marker': '^', 'color': 'green'},
                                   'permutation_fixed_blocks': {'marker': 'v', 'color': 'orange'},
                                   'top_k': {'marker': '>', 'color': 'blue'}}
    for dump in dumps_loaded:
        print("!" * 100)
        print(dump['params']['config']['algorithm_name'])
        print(dump['params']['config']['compressor_name'])
        print(dump['params']['config']['num_nodes'])
        print(dump['params']['config']['algorithm_master_params']['gamma_multiply'])
        print("Gamma: ", dump['params']['gamma'])
        i = unique_reg_paramterer.index(dump['params']['config'].get('reg_paramterer', 0.0))
        j = unique_split_with_controling_homogeneity.index(get_split_with_controling_homogeneity(dump))
        ax = axs[i][j]
        ax_2 = axs_2[i][j]
        ax_best = axs_best[i][j]
        
        function_values = np.array(dump['stat']['function_values'])
        args = [np.array(dump['stat']['max_bites_send_from_nodes']), function_values]
        color = best_experiments_to_markers[dump['params']['config']['compressor_name']]['color']
        if zero_lambda:
            markersize = 25
        else:
            markersize = 14
        kwargs = {'label' :"{}: Step size = {}".format(representation_name[dump['params']['config']['compressor_name']],
                                                       dump['params']['config']['algorithm_master_params']['gamma_multiply']),
                  'marker' : markers[processed_index[i][j]],
                  'markersize': markersize,
                  'color': color}
        if dump['_index'] not in use_experiment[i][j][dump['params']['config']['compressor_name']]:
            print("Skipping")
            continue
        if dump['params']['config']['compressor_name'] == 'rand_k' or dump['params']['config']['compressor_name'] == 'permutation':
            line, = ax.plot(*args, **kwargs)
            if dump['params']['config']['compressor_name'] == 'rand_k':
                line.set_linestyle('dotted')
            line.set_markevery(every=0.2)
        if dump['params']['config']['compressor_name'] == 'top_k' or dump['params']['config']['compressor_name'] == 'permutation':
            line, = ax_2.plot(*args, **kwargs)
            if dump['params']['config']['compressor_name'] == 'rand_k':
                line.set_linestyle('dashed')
            line.set_markevery(every=0.2)
            
        if best_experiment[i][j][dump['params']['config']['compressor_name']][0][1] == dump['_index']:
            kwargs['marker'] = best_experiments_to_markers[dump['params']['config']['compressor_name']]['marker']
            kwargs['color'] = color
            line, = ax_best.plot(*args, **kwargs)
            if dump['params']['config']['compressor_name'] == 'rand_k':
                line.set_linestyle('dotted')
            if dump['params']['config']['compressor_name'] == 'top_k':
                line.set_linestyle('dashed')
            line.set_markevery(every=0.2)
        processed_index[i][j] += 1
    
    pad = 20
    max_show = 1.
    for axs_ in [axs, axs_2, axs_best]:
        for i in range(len(axs_)):
            for j in range(len(axs_[i])):
                if i == 0:
                    if zero_lambda:
                        ss = 40
                    else:
                        ss = 25
                    capt = r'Probability ($\^p$) = {}'.format(float_to_str(unique_split_with_controling_homogeneity[j]))
                    axs_[i][j].annotate(capt, xy=(0.5, 1), xytext=(0, pad),
                                    xycoords='axes fraction', textcoords='offset points',
                                    size=ss, ha='center', va='baseline')
                if j == 0 and not zero_lambda:
                    axs_[i][j].annotate(r'$\lambda$ = {}'.format(float_to_str(unique_reg_paramterer[i])),
                                    xy=(0, 0.5), xytext=(-axs_[i][j].yaxis.labelpad - pad, 0),
                                    xycoords=axs_[i][j].yaxis.label, textcoords='offset points',
                                    size=25, ha='right', va='center', rotation = 90)
                if zero_lambda:
                    if j == 0:
                        axs_[i][j].set_ylabel(r'$f(x^k)$', size=40)
                    if i == len(axs_) - 1:
                        axs_[i][j].set_xlabel('#bits / n', size=45)
                else:
                    if j == 0:
                        axs_[i][j].set_ylabel(r'$f(x^k)$', size=24)
                    if i == len(axs_) - 1:
                        axs_[i][j].set_xlabel('#bits / n', size=22)
                # axs_[i][j].set_ylim(10**(log_index[i][j] - 1), max_show)
                axs_[i][j].set_yscale('log')
                if zero_lambda:
                    axs_[i][j].legend(fontsize=35, loc='upper right')
                    axs_[i][j].xaxis.set_tick_params(labelsize=30)
                    axs_[i][j].yaxis.set_tick_params(labelsize=30)
                    axs_[i][j].yaxis.set_tick_params(labelsize=30, which='minor')
                    axs_[i][j].xaxis.get_offset_text().set_size(30)
                else:
                    axs_[i][j].legend(fontsize=20, loc='upper right')
                    axs_[i][j].xaxis.set_tick_params(labelsize=18)
                    axs_[i][j].yaxis.set_tick_params(labelsize=16)
                    axs_[i][j].yaxis.set_tick_params(labelsize=16, which='minor')
                    axs_[i][j].xaxis.get_offset_text().set_size(18)
    
    # if zero_lambda:
        # fig_best.subplots_adjust(=1.0)
    fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(output_path + ".eps", bbox_inches="tight")
    fig_2.subplots_adjust(left=0.15, top=0.95)
    fig_2.savefig(output_path + "_2.eps", bbox_inches="tight")
    fig_best.subplots_adjust(left=0.15, top=0.95)
    fig_best.savefig(output_path + "_best.eps", bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dumps_paths', required=True, nargs='+')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--zero_lambda', action='store_true')

    args = parser.parse_args()
    plot_results(args.output_path, args.zero_lambda)


if __name__ == "__main__":
    main()