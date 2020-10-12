import random
import operator
from data_cars import cars_1, car_labels_1
from collections import Counter
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from cpu_impl import change_data, change_labels

cars_dict = {0: "Buying Price", 1: "Price of maintenance", 2: "Number of doors", 3: "Person Capacity",
                     4: "Size of luggage boot", 5: "Estimated Saftey"}


"""Decision Trees mithilfe von PyCuda"""

car_data = change_data(cars_1)
car_labels = change_labels(car_labels_1)

car_data = np.array(car_data).astype(np.float32)
car_labels = np.array(car_labels).astype(np.float32)

car_data_d = gpuarray.to_gpu(car_data)
car_labels_d = gpuarray.to_gpu(car_labels)

ker = SourceModule("""
""")

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value


class Internal_Node:
    def __init__(self,
                 feature,
                 branches,
                 value):
        self.feature = feature
        self.branches = branches
        self.value = value

"""
Ueberlegungen:
- Arrays in den Funktionen in GPUarrays umwandeln

"""
def gini(dataset):
    impurity = 1
    label_counts = Counter(dataset)
    for label in label_counts:
        prob_of_label = label_counts[label] / len(dataset)
        impurity -= prob_of_label ** 2
    return impurity


def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    for subset in split_labels:
        info_gain -= gini(subset) * len(subset) / len(starting_labels)
    return info_gain


def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets


def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


def build_tree(data, labels, value=""):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain < 0.00000001:
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def print_tree(node, question_dict, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + str(node.labels))
        return
    # Print the question at this node
    print(spacing + "Splitting on " + str(question_dict[node.feature]))
    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print(spacing + '--> Branch ' + str(node.branches[i].value) + ':')
        print_tree(node.branches[i], question_dict, spacing + "  ")


def classify(datapoint, tree):
    if isinstance(tree, Leaf):
        max = tree.labels[list(tree.labels)[0]]
        best = list(tree.labels)[0]
        for label in tree.labels:
            if tree.labels[label] > max:
                best = label
                max = tree.labels[label]
        return best
    value = datapoint[tree.feature]
    for branch in tree.branches:
        if branch.value == value:
            return classify(datapoint, branch)

ker = SourceModule("""
""")