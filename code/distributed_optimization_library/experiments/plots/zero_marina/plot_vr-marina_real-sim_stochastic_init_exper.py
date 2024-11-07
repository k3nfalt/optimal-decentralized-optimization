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
    return dump['params']['config']['algorithm_master_params']['gamma'] * dump['params']['config']['algorithm_master_params'].get('gamma_multiply', 1.0)

def folder_to_megabatchsize(dump):
    # folder = dump['_folder']
    # folder = folder.split('_')
    # return int(folder[11])
    return 0

def plot_results(dumps_paths, output_path, plot_functions):
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
                    print("Let's try again: {}".format(dump))
                    time.sleep(0.5)
            if not completed_read:
                continue
            
            # if dump['params']['config']['algorithm_name'] == 'vr_marina' and dump['params']['gamma'] == 8:
            #     print(dump['stat']['norm_of_gradients'][-100:])
            #     asdasd
            
            # if len(dump['stat']['norm_of_gradients']) == 0:
            #     continue
            if dump['params']['config']['algorithm_name'] in ['marina_stochastic']:
                continue
            if dump['params']['config']['algorithm_name'] == 'zero_marina_stochastic':
                if dump['params']['config'].get('version', '') in ['underestimated_momentum', 'underestimated_all']:
                    continue
                print(dump['params']['config'].get('version', ''))
                dump['params']['config']['algorithm_name'] += str(dump['params']['config'].get('version', ''))
            dump['_folder'] = os.path.basename(dumps_path)
            unique_num_nodes.add(folder_to_megabatchsize(dump))
            dump['params']['config']['noise_lambda'] = 0.0
            unique_noise_lambda.add(dump['params']['config']['compressor_params']['number_of_coordinates'])
            dumps_loaded.append(dump)
    unique_num_nodes = sorted(list(unique_num_nodes))
    unique_noise_lambda = sorted(list(unique_noise_lambda))
    
    # if len(nodes) == 3:
    #     fig, axs = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    #     fig_2, axs_2 = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    #     fig_best, axs_best = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 12))
    # else:
    print(unique_num_nodes, unique_noise_lambda)
    fig, axs = plt.subplots(len(unique_num_nodes), len(unique_noise_lambda), figsize=(40, 20))
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

    representation_name = {'marina': 'MARINA',
                           'zero_marina': 'ZeroMARINA',
                           'vr_marina': 'VRMARINA',
                           'zero_marina_page': 'ZeroMARINAPage',
                           'marina_stochastic': 'VR-MARINA (online)',
                           'zero_marina_stochastic': 'MVR-',
                        #    'zero_marina_stochastic100000': 'MVR- (100000)',
                           'zero_marina_sync_stochastic': 'MVR-SYNC-',
                           'stochastic_gradient_descent': 'stochastic_gradient_descent'}
    
    def get_representation_name(name):
        if 'zero_marina_stochastic' in name:
            num = name[len('zero_marina_stochastic'):]
            return "MVR {}".format(num)
        else:
            return representation_name[name]
    
    index = 0
    for dump in dumps_loaded:
        dump['_index'] = index
        i = unique_num_nodes.index(folder_to_megabatchsize(dump))
        j = unique_noise_lambda.index(dump['params']['config']['compressor_params']['number_of_coordinates'])
        dd = len(dump['stat']['function_values'])
        # min_norm_gradient = np.nanmean(np.log10(dump['stat']['norm_of_gradients'])[int(0.1 * dd) : ])
        # min_norm_gradient = np.nanmean(np.log10(dump['stat']['function_values'])[int(0.1 * dd) : ])
        min_norm_gradient = np.nanmean(np.log10(dump['stat']['function_values'])[int(0.01 * dd) : int(0.1 * dd)])
        algorithm_name = dump['params']['config']['algorithm_name']
        if algorithm_name not in best_experiment[i][j]:
            best_experiment[i][j][algorithm_name] = []
        # if not np.isnan(dump['stat']['norm_of_gradients']).any() and not np.isinf(dump['stat']['norm_of_gradients']).any():
        best_experiment[i][j][algorithm_name].append((min_norm_gradient, dump['_index']))
        index += 1
    top_show = 6
    use_experiment = [[defaultdict(set) for _ in range(len(unique_noise_lambda))] for _ in range(len(unique_num_nodes))]
    for i in range(len(best_experiment)):
        for j in range(len(best_experiment[0])):
            for comp in best_experiment[i][j]:
                best_experiment[i][j][comp] = sorted(best_experiment[i][j][comp])
                for k in range(min(len(best_experiment[i][j][comp]), top_show)):
                    use_experiment[i][j][comp].add(best_experiment[i][j][comp][k][1])
    dumps_loaded = sorted(dumps_loaded, 
                          key=lambda dump: (dump['params']['config']['algorithm_name'],
                                            get_gamma(dump)))
    
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
    best_experiments_to_markers = {'marina': {'marker': '<', 'color': 'red'},
                                   'zero_marina': {'marker': '^', 'color': 'green'},
                                   'vr_marina': {'marker': '<', 'color': 'red'},
                                   'zero_marina_page': {'marker': '^', 'color': 'green'},
                                   'marina_stochastic': {'marker': '<', 'color': 'red'},
                                   'zero_marina_stochastic': {'marker': '^', 'color': 'green'},
                                   'zero_marina_sync_stochastic': {'marker': '>', 'color': 'blue'},
                                   'stochastic_gradient_descent': {'marker': '>', 'color': 'orange'}}
    
    num_to_marker = {}
    available_markers = iter([{'marker': '<', 'color': 'purple'}, 
                              {'marker': '>', 'color': 'orange'},
                              {'marker': '<', 'color': 'red'},
                              {'marker': 'x', 'color': 'yellow'},
                              {'marker': 'v', 'color': 'pink'}])
    def get_best_experiments_to_markers(name):
        if 'zero_marina_stochastic' in name:
            num = name[len('zero_marina_stochastic'):]
            if num not in num_to_marker:
                num_to_marker[num] = next(available_markers)
            return num_to_marker[num]
        else:
            return best_experiments_to_markers[name]
    
    for dump in dumps_loaded:
        print("!" * 100)
        print(dump['params']['config']['algorithm_name'])
        # print(dump['params']['config']['compressor_name'])
        print(dump['params']['config']['num_nodes'])
        print(dump['params']['config']['noise_lambda'])
        # print(dump['params']['config']['algorithm_master_params']['gamma_multiply'])
        print("Step size: ", get_gamma(dump))
        i = unique_num_nodes.index(folder_to_megabatchsize(dump))
        j = unique_noise_lambda.index(dump['params']['config']['compressor_params']['number_of_coordinates'])
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
        smoothed_func_values = moving_average(norm_of_gradients, 100)
        # smoothed_func_values = dump['stat']['norm_of_gradients']
        # smoothed_func_values = norm_of_gradients
        print(np.nanmin(norm_of_gradients))
        print(get_gamma(dump), dump['params']['config'].get('algorithm_master_params', {}).get('noise_momentum', None),
              dump['params']['config'].get('algorithm_master_params', {}).get('mega_batch_size', None))
        if np.isnan(smoothed_func_values).any() or np.isinf(smoothed_func_values).any():
            print("Skipping")
            continue
        # print(min_values[i][j])
        min_value = np.nanmin(norm_of_gradients)
        if 'batch_size' in dump['params']['config']['algorithm_master_params']:
            m_part = float(dump['params']['config']['algorithm_master_params']['batch_size']) / (dump['params']['config']['algorithm_master_params']['batch_size'] + 600000 / dump['params']['config']['num_nodes'])
            # prob_part = dump['params']['config']['compressor_params']['number_of_coordinates'] / float(20958)
            prob_part = dump['params']['config']['compressor_params']['number_of_coordinates'] / float(90000)
            omega = 1 / prob_part
            print("m_part: {} prob_part: {} omega: {}".format(m_part, prob_part, omega))
        # subindices = np.arange(0, len(norm_of_gradients), 1)
        # y = np.array(dump['stat']['max_bites_send_from_nodes'])[subindices]
        y = np.array(dump['stat']['max_bites_send_from_nodes'])
        # x = smoothed_func_values[subindices]
        x = smoothed_func_values
        print("Num: ", len(x))
        # mask = y <= 10**(min_power - 1)
        # y = y[mask]
        # x = x[mask]
        args = [y, x]
        color = get_best_experiments_to_markers(dump['params']['config']['algorithm_name'])['color']
        # if dump['params']['config']['algorithm_name'] == 'marina_stochastic':
        #     kwargs = {'label' :"{}: x{} MBS:{} Parallel: {} NC:{}".format(representation_name[dump['params']['config']['algorithm_name']],
        #                                         get_gamma(dump),
        #                                         dump['params']['config'].get('algorithm_master_params', {}).get('mega_batch_size', None),
        #                                         dump['params']['config'].get('parallel', False),
        #                                         dump['params']['config'].get('compressor_params', {}).get('number_of_coordinates', 'all')),
        #               'marker' : markers[processed_index[i][j]],
        #               'markersize': 25,
        #               'linewidth': 6,
        #               'color': color}
        # else:
        # kwargs = {'label' :"{}: x{} NM:{} Parallel: {} NC:{}".format(representation_name[dump['params']['config']['algorithm_name']],
        #                                     get_gamma(dump),
        #                                     dump['params']['config'].get('algorithm_master_params', {}).get('noise_momentum', None),
        #                                     dump['params']['config'].get('parallel', False),
        #                                     dump['params']['config'].get('compressor_params', {}).get('number_of_coordinates', 'all')),
        #         'marker' : markers[processed_index[i][j]],
        #         'linewidth': 6,
        #         'markersize': 25,
        #         'color': color}
        # kwargs = {'label' :"{}: Step size: {} Extra parameter: {}".format(representation_name[dump['params']['config']['algorithm_name']],
        #                                               get_gamma(dump),
        #                                               (dump['params']['config']['algorithm_master_params'].get('mega_batch_size', False) or 
        #                                                dump['params']['config']['algorithm_master_params'].get('noise_momentum', False))),
        #         'marker' : markers[processed_index[i][j]],
        #         'linewidth': 6,
        #         'markersize': 25,
        #         'color': color}
        kwargs = {'label' :"{}: Step size: {}".format(get_representation_name(dump['params']['config']['algorithm_name']),
                                                      get_gamma(dump)),
                'marker' : markers[processed_index[i][j]],
                'linewidth': 6,
                'markersize': 35,
                'markeredgecolor': 'black',
                'markeredgewidth': 2.0,
                'color': color}
        if dump['_index'] not in use_experiment[i][j][dump['params']['config']['algorithm_name']]:
            continue
        if plot_functions:
            min_power = -6
        else:
            min_power = -6
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
        # if best_experiment[i][j][dump['params']['config']['algorithm_name']][0][1] == dump['_index']:
        #     kwargs['marker'] = best_experiments_to_markers[dump['params']['config']['algorithm_name']]['marker']
        #     # kwargs['color'] = best_experiments_to_markers[dump['params']['config']['compressor_name']]['color']
        #     kwargs['color'] = color
        #     line, = ax_best.plot(*args, **kwargs)
        #     if dump['params']['config']['algorithm_name'] == 'marina':
        #         line.set_linestyle('dotted')
        #     if dump['params']['config']['algorithm_name'] == 'zero_marina':
        #         line.set_linestyle('dashed')
        #     line.set_markevery(every=0.2)
        processed_index[i][j] += 1
    
    pad = 20
    if plot_functions:
        max_show = 10.0
    else:
        # max_show = 1e-3
        # max_show = 10000000.0
        max_show = 1e-2
    for axs_ in [axs, axs_best, axs_2]:
        for i in range(len(axs_)):
            for j in range(len(axs_[i])):
                if i == 0:
                    ss = 40
                    capt = r'$K$ = {}'.format(unique_noise_lambda[j])
                    axs_[i][j].annotate(capt, xy=(0.5, 1), xytext=(0, pad),
                                    xycoords='axes fraction', textcoords='offset points',
                                    size=ss, ha='center', va='baseline')
                if j == 0:
                    # axs_[i][j].annotate(r'$\frac{1}{n \varepsilon}$ = {}'.format(unique_num_nodes[i]), 
                    axs_[i][j].annotate(r'$\frac{\sigma^2}{n \varepsilon}$ = ' + str(unique_num_nodes[i]), 
                                    xy=(0, 0.5), xytext=(-axs_[i][j].yaxis.labelpad - pad, 0),
                                    xycoords=axs_[i][j].yaxis.label, textcoords='offset points',
                                    size=40, ha='right', va='center', rotation = 90)
                    if plot_functions:
                        axs_[i][j].set_ylabel(r'$f(x^k) - f(x^*)$', size=18)
                    else:
                        axs_[i][j].set_ylabel(r'$||\nabla f(x^k)||^2$', size=40)
                axs_[i][j].set_xlabel('#bits / n', size=40)
                axs_[i][j].set_ylim(10**(log_index[i][j] - 1), max_show)
                axs_[i][j].set_yscale('log')
                axs_[i][j].legend(fontsize=30, loc='upper right', framealpha=0.5)
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

    args = parser.parse_args()
    plot_results(args.dumps_paths, args.output_path, args.functions)


if __name__ == "__main__":
    main()