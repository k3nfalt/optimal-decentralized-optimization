import argparse
import os
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def moving_average(x, w):
    w = np.ones((w,)) / w
    return scipy.ndimage.convolve1d(x, w)

def plot_results(dumps_paths, output_path, plot_functions, nodes):
    dumps_loaded = []
    unique_num_nodes = set()
    unique_noise_lambda = set()
    for dumps_path in dumps_paths:
        dumps = os.listdir(dumps_path)
        
        for dump in dumps:
            if dump == "source_folder":
                continue
            dump = json.load(open(os.path.join(dumps_path, dump)))
            if dump['params']['config']['num_nodes'] not in nodes:
                continue
            unique_num_nodes.add(dump['params']['config']['num_nodes'])
            unique_noise_lambda.add(dump['params']['config']['noise_lambda'])
            dumps_loaded.append(dump)
    unique_num_nodes = sorted(list(unique_num_nodes))
    unique_noise_lambda = sorted(list(unique_noise_lambda))
    
    if len(nodes) == 3:
        fig, axs = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
        fig_2, axs_2 = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
        fig_best, axs_best = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    else:
        fig, axs = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 20))
        fig_2, axs_2 = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 20))
        fig_best, axs_best = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 20))
    if str(type(axs[0])) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
        axs = [axs]
        axs_best = [axs_best]
        axs_2 = [axs_2]
    
    min_values = [[None] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    liptchist_constant = [[None] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    min_max_sv = [None] * len(unique_noise_lambda)
    best_experiment = [[{} for _ in range(len(unique_noise_lambda))] for _ in range(len(unique_num_nodes))]

    representation_name = {'rand_k': 'RandK',
                           'permutation': 'PermK',
                           'nodes_permutation': 'PermK',
                           'permutation_fixed_blocks': 'PermK (Fixed Blocks)',
                           'top_k': 'TopK'}
    
    index = 0
    for dump in dumps_loaded:
        dump['_index'] = index
        i = unique_num_nodes.index(dump['params']['config']['num_nodes'])
        j = unique_noise_lambda.index(dump['params']['config']['noise_lambda'])
        with open(os.path.join(dump['params']['config']['dump_path'], "function_stats.json")) as fd:
            function_stats = json.load(fd)
        if min_values[i][j] is None:
            min_values[i][j] = function_stats['min_value']
            liptchist_constant[i][j] = function_stats['liptschitz_gradient_constant']
        else:
            np.testing.assert_almost_equal(min_values[i][j], function_stats['min_value'])
            np.testing.assert_almost_equal(liptchist_constant[i][j], function_stats['liptschitz_gradient_constant'])
        if min_max_sv[j] is None:
            min_max_sv[j] = [function_stats['smoothness_variance_bound'], function_stats['smoothness_variance_bound']]
        min_max_sv[j][0] = min(min_max_sv[j][0], function_stats['smoothness_variance_bound'])
        min_max_sv[j][1] = max(min_max_sv[j][1], function_stats['smoothness_variance_bound'])
        dd = len(dump['stat']['norm_of_gradients'])
        min_norm_gradient = np.nanmean(np.log10(dump['stat']['norm_of_gradients'])[int(0.1 * dd) : ])
        compressor_name = dump['params']['config']['compressor_name']
        if compressor_name not in best_experiment[i][j]:
            best_experiment[i][j][compressor_name] = []
        if not np.isnan(dump['stat']['norm_of_gradients']).any() and not np.isinf(dump['stat']['norm_of_gradients']).any():
            best_experiment[i][j][compressor_name].append((min_norm_gradient, dump['_index']))
        else:
            print(compressor_name, dump['params']['config']['num_nodes'], dump['params']['config']['noise_lambda'], dump['params']['config']['algorithm_master_params']['gamma_multiply'], 'Diverged')
        index += 1
    top_show = 100
    use_experiment = [[defaultdict(set) for _ in range(len(unique_noise_lambda))] for _ in range(len(unique_num_nodes))]
    for i in range(len(best_experiment)):
        for j in range(len(best_experiment[0])):
            for comp in best_experiment[i][j]:
                best_experiment[i][j][comp] = sorted(best_experiment[i][j][comp])
                for k in range(min(len(best_experiment[i][j][comp]), top_show)):
                    use_experiment[i][j][comp].add(best_experiment[i][j][comp][k][1])
    dumps_loaded = sorted(dumps_loaded, 
                          key=lambda dump: (dump['params']['config']['compressor_name'],
                                            dump['params']['config']['algorithm_master_params']['gamma_multiply']))
    
    log_index = [[0] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    processed_index = [[0] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    markers = ['v','^','<','>','s','p','P','*','h','H','+','x','X','D','d','|','_']
    markers += markers
    markers += markers
    colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors += colors
    colors += colors
    best_experiments_to_markers = {'rand_k': {'marker': '<', 'color': 'red'},
                                   'permutation': {'marker': '^', 'color': 'green'},
                                   'nodes_permutation': {'marker': '^', 'color': 'green'},
                                   'permutation_fixed_blocks': {'marker': 'v', 'color': 'orange'},
                                   'top_k': {'marker': '>', 'color': 'blue'}}
    for dump in dumps_loaded:
        if dump['params']['config']['algorithm_master_params']['gamma_multiply'] < 0.99:
            continue
        print("!" * 100)
        print(dump['params']['config']['algorithm_name'])
        print(dump['params']['config']['compressor_name'])
        print(dump['params']['config']['num_nodes'])
        print(dump['params']['config']['noise_lambda'])
        print(dump['params']['config']['algorithm_master_params']['gamma_multiply'])
        print("Gamma: ", dump['params']['gamma'])
        i = unique_num_nodes.index(dump['params']['config']['num_nodes'])
        j = unique_noise_lambda.index(dump['params']['config']['noise_lambda'])
        ax = axs[i][j]
        ax_2 = axs_2[i][j]
        ax_best = axs_best[i][j]
        if False:
            func_values = np.array(dump['stat']['function_values'])
            smoothed_func_values = moving_average(func_values, 100)
            func_values_minus_min_values = func_values - min_values[i][j]
            smoothed_func_values_minus_min_values = smoothed_func_values - min_values[i][j]
            print(np.nanmin(func_values))
            if np.isnan(func_values).any() or np.nanmin(func_values) > 0.01:
                print("Skipping")
                continue
            print(min_values[i][j])
            print(dump['params']['config']['dump_path'])
            min_value = np.nanmin(func_values_minus_min_values)
            if min_value <= 10**(-7):
                log_index[i][j] = -7
            else:
                log_index[i][j] = min(log_index[i][j], int(np.log10(min_value)))
            # line, = ax.plot(np.array(dump['stat']['bites_send_from_nodes']) / dump['params']['config']['num_nodes'], 
            line, = ax.plot(np.array(dump['stat']['max_bites_send_from_nodes']), 
                            smoothed_func_values_minus_min_values,
                    label="{}: x{}".format(
                        representation_name[dump['params']['config']['compressor_name']],
                        dump['params']['config']['algorithm_master_params']['gamma_multiply']),
                    marker=markers[processed_index[i][j]])
        else:
            if plot_functions:
                norm_of_gradients = np.array(dump['stat']['function_values']) - min_values[i][j]
            else:
                norm_of_gradients = np.array(dump['stat']['norm_of_gradients']) ** 2
            # smoothed_func_values = moving_average(norm_of_gradients, 1)
            smoothed_func_values = norm_of_gradients
            print(np.nanmin(norm_of_gradients))
            if np.isnan(smoothed_func_values).any() or np.isinf(smoothed_func_values).any():
                print("Skipping")
                continue
            print(min_values[i][j])
            print(dump['params']['config']['dump_path'])
            min_value = np.nanmin(norm_of_gradients)
            if plot_functions:
                min_power = -6
            else:
                min_power = -8
            if min_value <= 10**(min_power):
                log_index[i][j] = min_power
            else:
                log_index[i][j] = min(log_index[i][j], int(np.log10(min_value)))
            subindices = np.arange(0, len(norm_of_gradients), 100)
            y = np.array(dump['stat']['max_bites_send_from_nodes'])[subindices]
            x = smoothed_func_values[subindices]
            mask = y <= 10**(min_power - 1)
            # y = y[mask]
            # x = x[mask]
            args = [y, x]
            color = best_experiments_to_markers[dump['params']['config']['compressor_name']]['color']
            if (dump['params']['config']['compressor_name'] == 'rand_k'):
                if dump['params']['config']['algorithm_master_params']['optimal_scale']:
                    color = 'green'
            kwargs = {'label' :"{}: x{}".format(representation_name[dump['params']['config']['compressor_name']],
                                                dump['params']['config']['algorithm_master_params']['gamma_multiply']),
                      'marker' : markers[processed_index[i][j]],
                      'markersize': 14,
                      'color': color}
                    #   'color': colors[processed_index[i][j]]}
            if dump['_index'] not in use_experiment[i][j][dump['params']['config']['compressor_name']]:
                continue
            if dump['params']['config']['compressor_name'] != 'top_k':
                line, = ax.plot(*args, **kwargs)
                if dump['params']['config']['compressor_name'] == 'rand_k':
                    line.set_linestyle('dotted')
                if dump['params']['config']['compressor_name'] == 'top_k':
                    line.set_linestyle('dashed')
                line.set_markevery(every=0.2)
            if dump['params']['config']['compressor_name'] != 'rand_k':
                line, = ax_2.plot(*args, **kwargs)
                if dump['params']['config']['compressor_name'] == 'rand_k':
                    line.set_linestyle('dotted')
                if dump['params']['config']['compressor_name'] == 'top_k':
                    line.set_linestyle('dashed')
                line.set_markevery(every=0.2)
            if best_experiment[i][j][dump['params']['config']['compressor_name']][0][1] == dump['_index']:
                kwargs['marker'] = best_experiments_to_markers[dump['params']['config']['compressor_name']]['marker']
                # kwargs['color'] = best_experiments_to_markers[dump['params']['config']['compressor_name']]['color']
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
    for axs_ in [axs, axs_best, axs_2]:
        for i in range(len(axs_)):
            for j in range(len(axs_[i])):
                if i == 0:
                    lc = np.array(liptchist_constant)
                    lc = round(float(np.mean(lc[:, j])), 2)
                    if len(nodes) == 3:
                        if j == 0:
                            capt = r'$L_\pm$ = 0; $L_-$ = {}'.format(lc)
                        else:
                            sv = round(float(np.mean(min_max_sv[j])), 2)
                            # capt = r'$L_\pm$ $\approx$ {}; $L_-$ $\approx$ {}'.format(sv, lc)
                            capt = r'$L_\pm$ = {}; $L_-$ = {}'.format(sv, lc)   
                    else:
                        if j == 0:
                            capt = r'NS = 0; $L_\pm$ = 0; $L_-$ $\approx$ {}'.format(lc)
                        else:
                            sv = round(float(np.mean(min_max_sv[j])), 2)
                            capt = r'NS = {}; $L_\pm$ $\approx$ {}; $L_-$ $\approx$ {}'.format(
                                unique_noise_lambda[j], sv, lc)
                    axs_[i][j].annotate(capt, xy=(0.5, 1), xytext=(0, pad),
                                    xycoords='axes fraction', textcoords='offset points',
                                    size=25, ha='center', va='baseline')
                if j == 0:
                    axs_[i][j].annotate('Number of nodes: {}'.format(unique_num_nodes[i]), 
                                    xy=(0, 0.5), xytext=(-axs_[i][j].yaxis.labelpad - pad, 0),
                                    xycoords=axs_[i][j].yaxis.label, textcoords='offset points',
                                    size=20, ha='right', va='center', rotation = 90)
                if plot_functions:
                    axs_[i][j].set_ylabel(r'$f(x^k) - f(x^*)$', size=18)
                else:
                    axs_[i][j].set_ylabel(r'$||\nabla f(x^k)||^2$', size=18)
                axs_[i][j].set_xlabel('#bits / n', size=20)
                axs_[i][j].set_ylim(10**(log_index[i][j] - 1), max_show)
                axs_[i][j].set_yscale('log')
                axs_[i][j].legend(fontsize=20, loc='upper right')
                axs_[i][j].xaxis.set_tick_params(labelsize=16)
                axs_[i][j].yaxis.set_tick_params(labelsize=17)
                axs_[i][j].xaxis.get_offset_text().set_size(18)
    
    # fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(output_path + ".eps", bbox_inches="tight")
    fig_best.subplots_adjust(left=0.15, top=0.95)
    fig_best.savefig(output_path + "_best.eps", bbox_inches="tight")
    fig_2.subplots_adjust(left=0.15, top=0.95)
    fig_2.savefig(output_path + "_2.eps", bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_paths', required=True, nargs='+')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--functions', action='store_true')
    parser.add_argument('--nodes', nargs='+', default=[10, 100, 1000, 10000], type=int)

    args = parser.parse_args()
    plot_results(args.dumps_paths, args.output_path, args.functions, args.nodes)


if __name__ == "__main__":
    main()