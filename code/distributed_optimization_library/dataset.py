import os

import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
import scipy.sparse


class Dataset(object):
    def __init__(self, data, labels):
        self._data = np.copy(data)
        self._labels = np.copy(labels)
        
    def copy(self):
        return Dataset(self._data, self._labels)
    
    def __len__(self):
        return len(self._data)
    
    def split(self, number_of_parts, ignore_remainder=False):
        data = self._data
        labels = self._labels
        if ignore_remainder:
            number_of_samples_without_remainder = (len(labels) // number_of_parts) * number_of_parts
            data = data[:number_of_samples_without_remainder, :]
            labels = labels[:number_of_samples_without_remainder]
        assert number_of_parts <= len(labels)
        data_list = np.array_split(data, number_of_parts)
        labels_list = np.array_split(labels, number_of_parts)
        return [Dataset(data, labels) for data, labels in zip(data_list, labels_list)]

    @staticmethod
    def combine(datasets):
        data = np.concatenate([dataset._data for dataset in datasets], axis=0)
        labels = np.concatenate([dataset._labels for dataset in datasets], axis=0)
        return Dataset(data, labels)

    def number_of_classes(self):
        return len(np.unique(self._labels))
    
    def get_data_and_labels(self):
        return self._data, self._labels
    
    def shuffle(self, seed):
        generator = np.random.default_rng(seed=seed)
        random_permutation = generator.permutation(len(self._labels))
        self._labels = self._labels[random_permutation]
        self._data = self._data[random_permutation, :]
        
    def split_into_groups_by_labels(self, number_of_parts):
        indices_sorted = np.argsort(self._labels, kind='stable')
        labels = self._labels[indices_sorted]
        data = self._data[indices_sorted, :]
        unique_labels = np.unique(labels)
        assert number_of_parts % len(unique_labels) == 0
        number_of_parts_per_class = number_of_parts // len(unique_labels)
        datasets = []
        nodes_indices_splits = [0]
        for label in unique_labels:
            mask = labels == label
            labels_mask = labels[mask]
            data_mask = data[mask, :]
            dataset = Dataset(data_mask, labels_mask)
            dataset_split = dataset.split(number_of_parts_per_class)
            datasets += dataset_split
            nodes_indices_splits.append(nodes_indices_splits[-1] + len(dataset_split))
        return datasets, nodes_indices_splits

    def split_with_controling_homogeneity(self, number_of_parts, prob_taking_from_hold_out, seed):
        datasets = self.split(number_of_parts + 1)
        hold_out_dataset = datasets[0]
        datasets = datasets[1:]
        generator = np.random.default_rng(seed=seed)
        for dataset in datasets:
            number_of_samples = len(dataset._labels)
            indices_taking_from_hold_out = np.where(generator.random(size=(number_of_samples,)) 
                                                    <= prob_taking_from_hold_out)[0]
            dataset._labels[indices_taking_from_hold_out] = hold_out_dataset._labels[indices_taking_from_hold_out]
            dataset._data[indices_taking_from_hold_out, :] = hold_out_dataset._data[indices_taking_from_hold_out, :]
        return datasets
    
    def split_with_all_dataset(self, number_of_parts, prob_taking_all_dataset, seed,
                               max_number=1000):
        assert prob_taking_all_dataset > 0.0
        datasets = self.split(number_of_parts)
        data = self._data[:max_number, :]
        labels = self._labels[:max_number]
        for index in range(number_of_parts):
            if float(index) / number_of_parts < prob_taking_all_dataset:
                datasets[index] = Dataset(data, labels)
        return datasets

    def equalize_to_same_number_samples_per_class(self, number_samples):
        unique_labels = np.unique(self._labels)
        indices_to_keep = []
        for label in unique_labels:
            indices = np.where(self._labels == label)[0]
            assert number_samples <= len(indices)
            indices = indices[:number_samples]
            indices_to_keep.append(indices)
        indices_to_keep = np.concatenate(indices_to_keep)
        indices_to_keep = np.sort(indices_to_keep)
        equalized_labels = self._labels[indices_to_keep]
        equalized_data = self._data[indices_to_keep, :]
        return Dataset(equalized_data, equalized_labels)
    
    def subsample_classes(self, number_of_classes, seed):
        unique_labels = np.unique(self._labels)
        assert number_of_classes <= len(unique_labels)
        generator = np.random.default_rng(seed=seed)
        random_labels = generator.permutation(unique_labels)
        random_labels = np.sort(random_labels[:number_of_classes])
        mask = np.zeros(len(self), dtype=np.bool)
        remap_labels = -np.ones(len(self), dtype=np.int64)
        remap_label = 0
        for label in random_labels:
            mask_label = self._labels == label
            remap_labels[mask_label] = remap_label
            mask = np.logical_or(mask, mask_label)
            remap_label += 1
        return Dataset(self._data[mask,:], remap_labels[mask])


class SparseDataset(object):
    def __init__(self, data, labels):
        self._data = data.copy()
        self._labels = np.copy(labels)
    
    def split(self, number_of_parts, ignore_remainder=False):
        assert ignore_remainder
        data = self._data
        labels = self._labels
        number_of_samples_without_remainder = (len(labels) // number_of_parts) * number_of_parts
        data = data[:number_of_samples_without_remainder, :]
        labels = labels[:number_of_samples_without_remainder]
        assert number_of_parts <= len(labels)
        part_size = len(labels) // number_of_parts
        output_datasets = []
        for index in range(number_of_parts):
            first, last = part_size * index, part_size * (index + 1)
            output_datasets.append(SparseDataset(data[first:last, :], labels[first:last]))
        return output_datasets

    def number_of_classes(self):
        return len(np.unique(self._labels))
    
    def get_data_and_labels(self):
        return self._data.todense(), self._labels
    
    def shuffle(self, seed):
        generator = np.random.default_rng(seed=seed)
        random_permutation = generator.permutation(len(self._labels))
        self._labels = self._labels[random_permutation]
        self._data = self._data[random_permutation, :]


class LibSVMDataset(object):
    @staticmethod
    def from_file(path_to_folder, name, return_sparse=False, train=True):
        if name in ['epsilon_normalized_parsed', 'rcv1_test.binary_parsed']:
            assert train
            data = scipy.sparse.load_npz(os.path.join(path_to_folder, name, 'data.npz'))
            if not return_sparse:
                data = data.todense()
            labels = np.load(os.path.join(path_to_folder, name, 'labels.npy'))
        else:
            path_to_file = os.path.join(path_to_folder, name)
            if not train:
                path_to_file += '.t'
            data, labels = load_svmlight_file(path_to_file)
            data = data.toarray().astype(np.float32)
        print("Original labels: {}".format(np.unique(labels, return_counts=True)))
        print("Features: {}".format(data.shape))
        if name == 'mushrooms':
            labels = labels.astype(np.int64) - 1
        elif name == 'w8a':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'a9a':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'australian':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'covtype.libsvm.binary':
            labels = labels.astype(np.int64)
            labels -= 1
        elif name == 'madelon':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'real-sim':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'gisette_scale':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'epsilon_normalized' or name == 'epsilon_normalized_parsed':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'rcv1_test.binary_parsed':
            labels = labels.astype(np.int64)
            labels[labels == -1] = 0
        elif name == 'aloi':
            labels = labels.astype(np.int64)
        elif name == 'cifar10':
            data = data / 255.
            labels = labels.astype(np.int64)
        else:
            raise RuntimeError("Wrong dataset")
        if not return_sparse:
            return Dataset(data, labels)
        else:
            return SparseDataset(data, labels)


class MNISTDataset(Dataset):
    def __init__(self, path_to_dataset, train=True):
        file_name = "mnist_train.csv" if train else "mnist_test.csv"
        dataframe = pd.read_csv(os.path.join(path_to_dataset, file_name))
        labels = dataframe["label"].to_numpy(dtype=np.int64)
        print("Original labels: {}".format(np.unique(labels, return_counts=True)))
        data = dataframe.drop(labels = ["label"], axis = 1).to_numpy(dtype=np.float32)
        data = data / 255.0
        super(MNISTDataset, self).__init__(data, labels)
