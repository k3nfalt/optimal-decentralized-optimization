import argparse
import os
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.lines import Line2D
import time

def moving_average(x, w):
    w = np.ones((w,)) / w
    return scipy.ndimage.convolve1d(x, w)

def get_gamma(dump):
    return dump['params']['config']['algorithm_master_params']['gamma']


def plot_results(dumps_paths, output_path, plot_functions, ignore_methods, max_show):
    dumps_loaded = []
    unique_num_nodes = set()
    unique_noise_lambda = set()
    for dumps_path in dumps_paths:
        dumps = os.listdir(dumps_path)
        
        for dump in dumps:
            if dump == "source_folder":
                continue
            completed_read = False
            while not completed_read:
                try:
                    dump = json.load(open(os.path.join(dumps_path, dump)))
                    completed_read = True
                except json.decoder.JSONDecodeError as ex:
                    print("Let's try again")
                    time.sleep(5)
            
            unique_num_nodes.add(dump['params']['config']['num_nodes'])
            dump['params']['config']['noise_lambda'] = 0.0
            unique_noise_lambda.add(1)
            dumps_loaded.append(dump)
    unique_num_nodes = sorted(list(unique_num_nodes))
    unique_noise_lambda = sorted(list(unique_noise_lambda))
    
    # if len(nodes) == 3:
    #     fig, axs = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    #     fig_2, axs_2 = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    #     fig_best, axs_best = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    # else:
    fig, axs = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(20, 10))
    fig_2, axs_2 = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 20))
    fig_best, axs_best = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 20))
    # if str(type(axs[0])) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
    # axs = [axs]
    # axs_best = [axs_best]
    # axs_2 = [axs_2]
    axs = [[axs]]
    axs_best = [[axs_best]]
    axs_2 = [[axs_2]]
    
    min_values = [[None] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    liptchist_constant = [[None] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    min_max_sv = [None] * len(unique_noise_lambda)
    best_experiment = [[{} for _ in range(len(unique_noise_lambda))] for _ in range(len(unique_num_nodes))]

    representation_name = {'original_page': 'Vanilla PAGE',
                           'uniform_with_replacement': 'Uniform With Replacement',
                           'importance': 'Importance'}
    
    def get_representation_name(dump):
        sampling_name = dump['params']['config']['sampling_name']
        return representation_name[sampling_name]
    
    index = 0
    for dump in dumps_loaded:
        dump['_index'] = index
        i = unique_num_nodes.index(dump['params']['config']['num_nodes'])
        j = unique_noise_lambda.index(1)
        dd = len(dump['stat']['function_values'])
        # min_norm_gradient = np.nanmean(np.log10(dump['stat']['norm_of_gradients'])[int(0.1 * dd) : ])
        min_norm_gradient = np.nanmean(np.log10(dump['stat']['function_values'])[int(0.1 * dd) : ])
        sampling_name = dump['params']['config']['sampling_name']
        if sampling_name not in best_experiment[i][j]:
            best_experiment[i][j][sampling_name] = []
        # if not np.isnan(dump['stat']['norm_of_gradients']).any() and not np.isinf(dump['stat']['norm_of_gradients']).any():
        best_experiment[i][j][sampling_name].append((min_norm_gradient, dump['_index']))
        index += 1
    top_show = 5
    use_experiment = [[defaultdict(set) for _ in range(len(unique_noise_lambda))] for _ in range(len(unique_num_nodes))]
    for i in range(len(best_experiment)):
        for j in range(len(best_experiment[0])):
            for comp in best_experiment[i][j]:
                best_experiment[i][j][comp] = sorted(best_experiment[i][j][comp])
                for k in range(min(len(best_experiment[i][j][comp]), top_show)):
                    use_experiment[i][j][comp].add(best_experiment[i][j][comp][k][1])
    dumps_loaded = sorted(dumps_loaded, 
                          key=lambda dump: (dump['params']['config']['sampling_name'],
                                            -get_gamma(dump)),
                          reverse=True)
    
    log_index = [[0] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    processed_index = [[0] *len(unique_noise_lambda) for _ in range(len(unique_num_nodes))]
    markers = ['v','^','<','>','s','p','P','*','h','H','x','X','D','d']
    # markers = ['v','^','<','>','s','p','P','*','h','H','x','X','D','d']
    # num_markers = len(markers)
    # markers += markers
    # markers += markers
    # markers = list(Line2D.markers.keys())
    markers += markers
    markers += markers
    colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors += colors
    colors += colors
    # best_experiments_to_markers = {'rand_k': {'marker': '<', 'color': 'red'},
    #                                'permutation': {'marker': '^', 'color': 'green'},
    #                                'nodes_permutation': {'marker': '^', 'color': 'green'},
    #                                'permutation_fixed_blocks': {'marker': 'v', 'color': 'orange'},
    #                                'top_k': {'marker': '>', 'color': 'blue'}}
    best_experiments_to_markers = {'original_page': {'marker': '<', 'color': 'red'},
                                   'uniform_with_replacement': {'marker': '^', 'color': 'green'},
                                   'importance': {'marker': '>', 'color': 'blue'}}
    
    def get_marker(sampling_name):
        return best_experiments_to_markers[sampling_name]
    
    for dump in dumps_loaded:
        print("!" * 100)
        print("Sampling: {}".format(dump['params']['config']['sampling_name']))
        print("Number of iter: {}".format(dump['stat']['number_of_iterations']))
        print(dump['params']['config']['algorithm_name'])
        # print(dump['params']['config']['compressor_name'])
        print(dump['params']['config']['num_nodes'])
        print(dump['params']['config']['noise_lambda'])
        # print(dump['params']['config']['algorithm_master_params']['gamma_multiply'])
        print("Step size: ", get_gamma(dump))
        i = unique_num_nodes.index(dump['params']['config']['num_nodes'])
        j = unique_noise_lambda.index(1)
        ax = axs[i][j]
        ax_2 = axs_2[i][j]
        ax_best = axs_best[i][j]
        if plot_functions:
            # norm_of_gradients = np.array(dump['stat']['function_values']) - min_values[i][j]
            # norm_of_gradients = np.array(dump['stat']['gradient_estimator_error'])
            norm_of_gradients = np.array(dump['stat']['function_values'])
        else:
            norm_of_gradients = np.array(dump['stat']['norm_of_gradients']) ** 2
        # max_accuracy_train = np.max(dump['stat']['accuracy_train'])
        # smoothed_func_values = moving_average(norm_of_gradients, 1)
        smoothed_func_values = norm_of_gradients
        print(np.nanmin(norm_of_gradients))
        print(get_gamma(dump), dump['params']['config'].get('algorithm_master_params', {}).get('noise_momentum', None),
              dump['params']['config'].get('algorithm_master_params', {}).get('mega_batch_size', None))
        if np.isnan(smoothed_func_values).any() or np.isinf(smoothed_func_values).any():
            print("Skipping")
            continue
        # print(min_values[i][j])
        min_value = np.nanmin(norm_of_gradients)
        # subindices = np.arange(0, len(norm_of_gradients), 1)
        # y = np.array(dump['stat']['max_bites_send_from_nodes'])[subindices]
        # y = np.array(dump['stat']['max_bites_send_from_nodes'])
        # x = smoothed_func_values[subindices]
        x = smoothed_func_values
        # y = np.arange(len(x))
        # y = (y * (dump['stat']['number_of_iterations'] / len(y))).astype(np.int32)
        print("ESTIMATING COMPLEXITY")
        if dump['params']['config']['dataset_name'] =='w8a':
            num_functions = 49749
        if dump['params']['config']['dataset_name'] =='australian':
            num_functions = 690
        y = np.array([(dd['gradient'] * num_functions + dd['batch_gradient_at_points']) 
                      for dd in dump['stat']['statistics']])
        # mask = y <= 10**(min_power - 1)
        # y = y[mask]
        # x = x[mask]
        args = [y, x]
        color = get_marker(dump['params']['config']['sampling_name'])['color']
        # kwargs = {'label' :"{}: Step size: {}".format(get_representation_name(dump),
        #                                               get_gamma(dump)),
        #         'marker' : markers[processed_index[i][j]],
        #         'linewidth': 10,
        #         'markersize': 35,
        #         'markeredgecolor': 'black',
        #         'markeredgewidth': 2.0,
        #         'color': color}
        kwargs = {'label' :"{}".format(get_representation_name(dump)),
                'marker' : markers[processed_index[i][j]],
                'linewidth': 10,
                'markersize': 35,
                'markeredgecolor': 'black',
                'markeredgewidth': 2.0,
                'color': color}
        if dump['_index'] not in use_experiment[i][j][dump['params']['config']['sampling_name']]:
            continue
        if plot_functions:
            min_power = -8
        else:
            min_power = -8
        if min_value <= 10**(min_power):
            log_index[i][j] = min_power
        else:
            log_index[i][j] = min(log_index[i][j], int(np.log10(min_value)))
        line, = ax.plot(*args, **kwargs)
        if dump['params']['config']['algorithm_name'] == 'marina':
            line.set_linestyle('dotted')
        if dump['params']['config']['algorithm_name'] == 'zero_marina':
            line.set_linestyle('dashed')
        line.set_markevery(every=0.2)
        if 'original' in dump['params']['config']['sampling_name']:
            line.set_linestyle('dotted')
        if 'importance' in dump['params']['config']['sampling_name']:
            line.set_linestyle('dashed')
        if best_experiment[i][j][dump['params']['config']['sampling_name']][0][1] == dump['_index']:
            kwargs['marker'] = get_marker(dump['params']['config']['sampling_name'])['marker']
            # kwargs['color'] = best_experiments_to_markers[dump['params']['config']['compressor_name']]['color']
            kwargs['color'] = color
            line, = ax_best.plot(*args, **kwargs)
            if dump['params']['config']['algorithm_name'] == 'marina':
                line.set_linestyle('dotted')
            if dump['params']['config']['algorithm_name'] == 'zero_marina':
                line.set_linestyle('dashed')
            line.set_markevery(every=0.2)
        processed_index[i][j] += 1
    
    pad = 20
    # if plot_functions:
    #     max_show = 10.0
    # else:
    #     # max_show = 1e-3
    max_show = min(10 ** 6, max_show)
    for axs_ in [axs, axs_best, axs_2]:
        for i in range(len(axs_)):
            for j in range(len(axs_[i])):
                if i == 0:
                    ss = 40
                    # capt = r'$K$ = {}'.format(unique_noise_lambda[j])
                    # axs_[i][j].annotate(capt, xy=(0.5, 1), xytext=(0, pad),
                    #                 xycoords='axes fraction', textcoords='offset points',
                    #                 size=ss, ha='center', va='baseline')
                    llc_min = round(np.min(dump['params']['function_stats']['local_lipt']), 2)
                    llc_max = round(np.max(dump['params']['function_stats']['local_lipt']), 2)
                    llc_mean = round(np.mean(dump['params']['function_stats']['local_lipt']), 2)
                    llc_sqrt_mean = round(np.max(dump['params']['function_stats']['local_lipt']), 2)
                    capt = r'$\min\,L_i$ $=$ {}; $\max\,L_i$ $=$ {}'.format(llc_min, llc_max)
                    axs_[i][j].annotate(capt, xy=(0.5, 1), xytext=(0, pad),
                                    xycoords='axes fraction', textcoords='offset points',
                                    size=ss, ha='center', va='baseline')
                if j == 0:
                    axs_[i][j].annotate('n = {}'.format(dump['params']['function_stats']['number_of_functions']), 
                                    xy=(0, 0.5), xytext=(-axs_[i][j].yaxis.labelpad - pad, 0),
                                    xycoords=axs_[i][j].yaxis.label, textcoords='offset points',
                                    size=40, ha='right', va='center', rotation = 90)
                    if plot_functions:
                        axs_[i][j].set_ylabel(r'$f(x^k) - f(x^*)$', size=18)
                    else:
                        axs_[i][j].set_ylabel(r'$||\nabla f(x^k)||^2$', size=40)
                axs_[i][j].set_xlabel('# of gradient calculations', size=40)
                axs_[i][j].set_ylim(10**(log_index[i][j] - 1), max_show)
                axs_[i][j].set_yscale('log')
                axs_[i][j].legend(fontsize=40, loc='upper right', framealpha=0.5)
                axs_[i][j].xaxis.set_tick_params(labelsize=40)
                axs_[i][j].yaxis.set_tick_params(labelsize=40)
                axs_[i][j].xaxis.get_offset_text().set_size(40)
    
    # fig.tight_layout()
    fig.subplots_adjust(left=0.15, top=0.95)
    fig.savefig(output_path + ".pdf", bbox_inches="tight")
    # fig_best.subplots_adjust(left=0.15, top=0.95)
    # fig_best.savefig(output_path + "_best.eps", bbox_inches="tight")
    # fig_2.subplots_adjust(left=0.15, top=0.95)
    # fig_2.savefig(output_path + "_2.eps", bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_paths', required=True, nargs='+')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--functions', action='store_true')
    parser.add_argument('--ignore_methods', nargs="*")
    parser.add_argument('--max_show', type=float)

    args = parser.parse_args()
    plot_results(args.dumps_paths, args.output_path, args.functions, args.ignore_methods, args.max_show)


if __name__ == "__main__":
    main()