import argparse
import os
import json
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.ndimage
from scipy.interpolate import interp1d

def moving_average(x, w):
    w = np.ones((w,)) / w
    return scipy.ndimage.convolve1d(x, w)

def plot_results(dumps_paths, output_path, compressor_range):
    _, (ax_loss, ax_quality, ax_error) = plt.subplots(1, 3, figsize=(16, 8))
    for dumps_path in dumps_paths:
        dumps = os.listdir(dumps_path)
        dumps_loaded = []
        for dump in dumps:
            if dump == "source_folder":
                continue
            dump_json = json.load(open(os.path.join(dumps_path, dump)))
            dump_json['_path'] = dump
            # if dump_json['params']['config']['compressor_name'] in ['permutation', 'rand_k']:
            #     continue
            if np.isnan(dump_json['stat']['function_values']).any():
                print(dump_json['params']['config']['compressor_name'], 
                      dump_json['params']['config']['algorithm_master_params']['gamma_multiply'])
                print('Diverged')
                continue
            dumps_loaded.append(dump_json)
        
        def _key(dump):
            return (dump['params']['config']['compressor_name'],
                    dump['params']['config']['algorithm_master_params']['gamma_multiply'])
        dumps_loaded = sorted(dumps_loaded, key=_key)
        
        diverged_keys = []
        min_test = float('inf')
        for key, group in itertools.groupby(dumps_loaded, key=_key):
            if str(key[0]) in compressor_range and (key[1] >= compressor_range[key[0]][1] or key[1] < compressor_range[key[0]][0]):
                diverged_keys.append(key)
                continue
            min_ = None
            max_ = None
            # funcs_quality = []
            # funcs_loss = []
            # funcs_error = []
            # num_el = 0
            # found = False
            # for dump in group:
            #     x = np.array(dump['stat']['max_bites_send_from_nodes'])
            #     # x = np.array(dump['stat']['bites_send_from_nodes'])
            #     # x = range(len(dump['stat']['max_bites_send_from_nodes']))
            #     if len(x) < 2:
            #         continue
            #     found = True
            #     if min_ is None:
            #         min_ = min(x)
            #     min_ = max(min(x), min_)
            #     if max_ is None:
            #         max_ = max(x)
            #     max_ = min(max(x), max_)
            #     # y = moving_average(dump['stat']['accuracy_train'], 100)
            #     # y = np.array(dump['stat']['norm_of_gradients']) * 0.0
            #     y = np.array(dump['stat']['accuracy_test'])
            #     min_test = min(min_test, np.min(y))
            #     f = interp1d(x, y)
            #     funcs_quality.append(f)
            #     # if dump['params']['config']['compressor_name'] == 'permutation':
            #     #     print(dump['stat']['norm_of_gradients'])
            #     # y = moving_average(np.array(dump['stat']['norm_of_gradients']) ** 2, 100)
            #     # y = np.array(dump['stat']['norm_of_gradients']) ** 2
            #     # y = moving_average(np.array(dump['stat']['gradient_estimator_error']), 100)
            #     # y = np.array(dump['stat']['gradient_estimator_error'])
            #     # y = moving_average(np.array(dump['stat']['function_values']), 100)
            #     y = np.array(dump['stat']['function_values'])
            #     f = interp1d(x, y)
            #     funcs_loss.append(f)
            #     y = np.array(dump['stat']['gradient_estimator_error'])
            #     f = interp1d(x, y)
            #     funcs_error.append(f)
            #     num_el += 1
            #     print(dump['_path'])
            # if not found:
            #     continue
            # # assert num_el == 5
            # xnew = np.linspace(min_, max_, num=10000)
            # funcs_quality_values = []
            # funcs_loss_values = []
            # funcs_error_values = []
            # for index in range(len(funcs_quality)):
            #     funcs_quality_values.append(funcs_quality[index](xnew))
            #     funcs_loss_values.append(funcs_loss[index](xnew))
            #     funcs_error_values.append(funcs_error[index](xnew))
            # funcs_quality_values = np.stack(funcs_quality_values, axis=1)
            # y_quality = np.mean(funcs_quality_values, axis=1)
            # funcs_loss_values = np.stack(funcs_loss_values, axis=1)
            # y_loss = np.mean(funcs_loss_values, axis=1)
            # funcs_error_values = np.stack(funcs_error_values, axis=1)
            # y_error = np.mean(funcs_error_values, axis=1)
            
            group = list(group)
            assert len(group) == 1
            dump = group[0]
            xnew = np.array(dump['stat']['max_bites_send_from_nodes'])
            y_quality = np.array(dump['stat']['accuracy_test'])
            y_loss = np.array(dump['stat']['function_values'])
            if len(y_quality) == 0:
                y_quality = y_loss
            y_error = np.array(dump['stat']['gradient_estimator_error'])
            min_test = min(min_test, np.min(y_quality))
            
            print("!" * 100)
            print(dump['params']['config']['compressor_name'])
            print(dump['params']['config']['num_nodes'])
            # print("Num elem in group: {}".format(num_el))
            print("Gamma: ", dump['params']['gamma'])
            linestyle = None
            if dump['params']['config']['compressor_name'] == 'rand_k':
                linestyle = 'dotted'
            if dump['params']['config']['compressor_name'] == 'top_k':
                linestyle = 'dashed'
            if dump['params']['config']['compressor_name'] == 'group_permutation':
                linestyle = 'dashdot'
            line_quality, = ax_quality.plot(xnew, y_quality,
                    label="{} gamma_multiply: {}".format(
                        dump['params']['config']['compressor_name'],
                        dump['params']['config']['algorithm_master_params']['gamma_multiply']),
                    linestyle=linestyle)
            line_loss, = ax_loss.plot(xnew, y_loss,
                    label="{} gamma_multiply: {}".format(
                        dump['params']['config']['compressor_name'],
                        dump['params']['config']['algorithm_master_params']['gamma_multiply']),
                    linestyle=linestyle)
            line_error, = ax_error.plot(xnew, y_error,
                    label="{} gamma_multiply: {}".format(
                        dump['params']['config']['compressor_name'],
                        dump['params']['config']['algorithm_master_params']['gamma_multiply']),
                    linestyle=linestyle)
            # ax_loss.set_ylim(7 * 10**(-2), 10**(-1))
        ax_loss.set_ylabel(r'$||\nabla f(x^k)||^2$ (smoothed)')
        ax_loss.set_xlabel('#bits / n')
        ax_quality.set_ylabel(r'accuracy (smoothed)')
        ax_quality.set_xlabel('#bits / n. Test error: {}'.format(min_test))
            # ax_quality.set_ylim(0.8, 1)
        ax_loss.set_yscale('log')
        ax_error.set_yscale('log')
        # ax_quality.set_yscale('log')
    
    handles_loss, _ = ax_loss.get_legend_handles_labels()
    handles_quality, _ = ax_quality.get_legend_handles_labels()
    # for key in diverged_keys:
    #     patch = mpatches.Patch(color='black', 
    #                            label="{} gamma_multiply: {} DIVERGED".format(key[0], key[1]))
    #     handles_loss.append(patch) 
    #     handles_quality.append(patch) 
    
    ax_loss.legend(handles=handles_loss, fontsize=8)
    ax_quality.legend(handles=handles_quality, fontsize=8)
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_paths', required=True, nargs='+')
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--compressor_range')

    args = parser.parse_args()
    compressor_range = {}
    if args.compressor_range:
        compressor_range = json.loads(args.compressor_range)
    print(args.compressor_range)
    plot_results(args.dumps_paths, args.output_path, compressor_range)


if __name__ == "__main__":
    main()