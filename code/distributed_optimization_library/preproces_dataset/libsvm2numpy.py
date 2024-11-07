import argparse
import os
import numpy as np

from sklearn.datasets import load_svmlight_file
import scipy.sparse

def parse_dataset(dataset, path_to_folder):
    data, labels = load_svmlight_file(os.path.join(path_to_folder, dataset))
    output_folder = os.path.join(path_to_folder, dataset + '_parsed')
    os.mkdir(output_folder)
    data_path = os.path.join(output_folder, 'data.npz')
    labels_path = os.path.join(output_folder, 'labels.npy')
    scipy.sparse.save_npz(data_path, data.astype(np.float32))
    np.save(labels_path, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--path_to_dataset', required=True)
    args = parser.parse_args()
    parse_dataset(args.dataset, args.path_to_dataset)

if __name__ == "__main__":
    main()