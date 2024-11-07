import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def moving_average(x, w):
    w = np.ones((w,)) / w
    return scipy.ndimage.convolve1d(x, w)

def plot_results(dumps_paths, output_path):
    _, ax = plt.subplots()
    for dumps_path in dumps_paths:
        dumps = os.listdir(dumps_path)
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     # ax.plot(dump['stat']['bites_send_from_nodes'], dump['stat']['function_values'],
        #     #         label="gamma: {}".format(dump['params']['algorithm_master_params']['gamma']))
        #     print(dump['params']['config'])
        #     # print(dump['stat']['bites_send_from_nodes'][-1])
        #     ax.plot(np.array(dump['stat']['bites_send_from_nodes']) / dump['params']['config']['num_nodes'], dump['stat']['function_values'],
        #             label="{} number_of_coordinates: {}".format(
        #                 dump['params']['config']['compressor_name'],
        #                 dump['params']['config']['compressor_params']['number_of_coordinates'] if dump['params']['config']['compressor_name'] == 'rand_k' else 113 / 20))
        #     ax.set_ylabel('function_values')
        #     ax.set_xlabel('bites_send_from_nodes / num_nodes')
        #     ax.set_yscale('log')
        # ax.legend()
        # plt.savefig(output_path)
        
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     # ax.plot(dump['stat']['bites_send_from_nodes'], dump['stat']['function_values'],
        #     #         label="gamma: {}".format(dump['params']['algorithm_master_params']['gamma']))
        #     print(dump['params']['config'])
        #     ax.plot(range(len(dump['stat']['function_values'])), dump['stat']['function_values'],
        #             label="{} number_of_coordinates: {}".format(
        #                 dump['params']['config']['compressor_name'],
        #                 dump['params']['config']['compressor_params']['number_of_coordinates'] if dump['params']['config']['compressor_name'] == 'rand_k' else 113 / 20))
        #     ax.set_ylabel('function_values')
        #     ax.set_xlabel('bites_send_from_nodes / num_nodes')
        #     ax.set_yscale('log')
        # ax.legend()
        # plt.savefig(output_path)
        
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     # ax.plot(dump['stat']['bites_send_from_nodes'], dump['stat']['function_values'],
        #     #         label="gamma: {}".format(dump['params']['algorithm_master_params']['gamma']))
        #     print(dump['params']['config'])
        #     ax.plot(range(len(dump['stat']['function_values'])), dump['stat']['function_values'],
        #             label="{} number_of_coordinates: {} gamma_multiply: {}".format(
        #                 dump['params']['config']['compressor_name'],
        #                 dump['params']['config']['compressor_params']['number_of_coordinates'],
        #                 dump['params']['config']['algorithm_master_params']['gamma_multiply']))
        #     ax.set_ylabel('function_values')
        #     ax.set_xlabel('iteration number')
        #     ax.set_yscale('log')
        # ax.legend()
        # plt.savefig(output_path)
        
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     # ax.plot(dump['stat']['bites_send_from_nodes'], dump['stat']['function_values'],
        #     #         label="gamma: {}".format(dump['params']['algorithm_master_params']['gamma']))
        #     print(dump['params']['config'])
        #     ax.plot(range(len(dump['stat']['function_values'])), dump['stat']['function_values'],
        #             label="{} gamma_multiply: {}".format(
        #                 dump['params']['config']['compressor_name'],
        #                 dump['params']['config']['algorithm_master_params']['gamma_multiply']))
        #     ax.set_ylabel('function_values')
        #     ax.set_xlabel('iteration number')
        #     ax.set_yscale('log')
        # ax.legend()
        # plt.savefig(output_path)

        
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     print(dump['params']['config'])
        #     ax.plot(dump['stat']['bites_send_from_nodes'], dump['stat']['function_values'],
        #             label="{} number_of_coordinates: {} gamma_multiply: {}".format(
        #                 dump['params']['config']['compressor_name'],
        #                 dump['params']['config']['compressor_params']['number_of_coordinates'],
        #                 dump['params']['config']['algorithm_master_params']['gamma_multiply']))
        #     ax.set_ylabel('function_values')
        #     ax.set_xlabel('bites_send_from_nodes')
        #     ax.set_yscale('log')
        # ax.legend()
        # plt.savefig(output_path)
        
        # min_ = float('inf')
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     min_ = min(min_, np.min(np.array(dump['stat']['function_values'])))
        # min_ -= 1e-2
        
        dumps_loaded = []
        for dump in dumps:
            dump = json.load(open(os.path.join(dumps_path, dump)))
            dumps_loaded.append(dump)
        
        dumps_loaded = sorted(dumps_loaded, 
                              key=lambda dump: (dump['params']['config']['compressor_name'],
                                                dump['params']['config']['algorithm_master_params']['gamma_multiply']))
        
        min_ = 0
        
        for dump in dumps_loaded:
            # if dump['params']['config']['algorithm_master_params']['gamma_multiply'] < 200:
            #     continue
            print("!" * 100)
            print(dump['params']['config']['compressor_name'])
            print(dump['params']['config']['num_nodes'])
            # print(dump['params']['config']['noise_lambda'])
            print("Gamma: ", dump['params']['gamma'])
            # if dump['params']['config']['algorithm_master_params']['gamma_multiply'] > 1:
            #     continue
            if dump['params']['config']['num_nodes'] != 1000:
                continue
            # if dump['params']['config']['num_nodes'] != 10:
            #     continue
            # print(dump['params']['function_stats'])
            # print(dump['params']['config'])
            # print(dump['params']['config']['compressor_params'])
            # func_values = moving_average(np.array(dump['stat']['function_values']), 100)
            func_values = np.array(dump['stat']['function_values'])
            # mask = np.array(func_values) <= 1.0
            # print(func_values.shape)
            n = len(dump['stat']['bites_send_from_nodes'])
            
            # line, = ax.plot(np.array(dump['stat']['bites_send_from_nodes']) / dump['params']['config']['num_nodes'], 
            #                 func_values - min_,
            #         label="{} gamma_multiply: {}".format(
            #             dump['params']['config']['compressor_name'],
            #             dump['params']['config']['algorithm_master_params']['gamma_multiply']))
            
            line, = ax.plot(np.array(dump['stat']['bites_send_from_nodes']) / dump['params']['config']['num_nodes'], 
                            moving_average(dump['stat']['accuracy_train'], 100),
                    label="{} gamma_multiply: {}".format(
                        dump['params']['config']['compressor_name'],
                        dump['params']['config']['algorithm_master_params']['gamma_multiply']))
            
            if dump['params']['config']['compressor_name'] == 'rand_k':
                line.set_dashes([2, 2, 10, 2])
            # ax.set_ylabel('smoothed_function_values')
            # ax.set_ylim(2.5 * 10**(-1), 1)
            ax.set_ylim(0.8, 1)
            # ax.set_xlabel('bites_send_from_node; noise_lambda: {}; smoothness_variance_bound: {}'.format(
            #     dump['params']['config']['noise_lambda'], dump['params']['function_stats']['smoothness_variance_bound']))
        # ax.set_ylim(top=10**2)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        
        
        # for dump in dumps:
        #     dump = json.load(open(os.path.join(dumps_path, dump)))
        #     print(dump['params']['config'])
        #     print(dump['params']['config']['compressor_params'])
        #     # func_values = moving_average(np.array(dump['stat']['function_values']), 100)
        #     # mask = np.array(func_values) <= 1000.0
        #     # print(func_values.shape)
        #     if dump['params']['config']['num_nodes'] != 1000:
        #         continue
        #     func_values = np.array(dump['stat']['function_values'])
        #     line, = ax.plot(np.array(dump['stat']['bites_send_from_nodes']) / dump['params']['config']['num_nodes'], 
        #                     func_values,
        #             label="{} gamma_multiply: {}".format(
        #                 dump['params']['config']['compressor_name'],
        #                 dump['params']['config']['algorithm_master_params']['gamma_multiply']))
        #     if dump['params']['config']['compressor_name'] == 'rand_k':
        #         line.set_dashes([2, 2, 10, 2])
        #     ax.set_ylabel('smoothed_function_values')
        #     # ax.set_xlabel('bites_send_from_node; noise_lambda: {}; smoothness_variance_bound: {}'.format(
        #     #     dump['params']['config']['noise_lambda'], dump['params']['function_stats']['smoothness_variance_bound']))
        # # ax.set_ylim(top=10**2)
        # ax.set_yscale('log')
        # # ax.set_xscale('log')
        
    ax.legend(fontsize=5)
    plt.savefig(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dumps_paths', required=True, nargs='+')
    parser.add_argument('--output_path', required=True)

    args = parser.parse_args()
    plot_results(args.dumps_paths, args.output_path)


if __name__ == "__main__":
    main()