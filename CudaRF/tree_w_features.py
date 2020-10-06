import operator
from collections import Counter
import numpy as np
import random

np.random.seed(1)
random.seed(1)


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


def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


def build_tree_features(data, labels, value=""):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain < 0.1:
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        """Als Value wird hier das beste Feature zum Teilen gegeben, sehr Praktisch, da man dies in jeder Internal node dann 
        speichert was dazu fuehrt, dass man dieses rauspicken kann."""
        branch = build_tree_features(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def print_tree_features(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1: "Price of maintenance", 2: "Number of doors", 3: "Person Capacity",
                     4: "Size of luggage boot", 5: "Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + str(node.labels))
        return

    # Print the question at this node
    print(spacing + "Splitting on " + question_dict[node.feature])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print(spacing + '--> Branch ' + node.branches[i].value + ':')
        print_tree_features(node.branches[i], spacing + "  ")

def classify_feature(datapoint, tree):
    if isinstance(tree, Leaf):
        best = list(tree.labels)[0]
        max = tree.labels[list(tree.labels)[0]]
        for label in tree.labels:
            if tree.labels[label] > max:
                best = label
                max = tree.labels[label]
        return best
    value = datapoint[tree.feature]
    for branch in tree.branches:
        if branch.value == value:
            return classify_feature(datapoint, branch)

"""For Forest Creation:"""


def build_tree_forest(data, labels, value=""):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain < 0.00000001:
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree_forest(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def make_random_forest(n, training_data, training_labels):
    trees = []
    for i in range(n):
        indices = [random.randint(0, len(training_data) - 1) for x in range(len(training_data))]

        training_data_subset = [training_data[index] for index in indices]
        training_labels_subset = [training_labels[index] for index in indices]

        tree = build_tree_forest(training_data_subset, training_labels_subset)
        trees.append(tree)
    return trees

def change_labels(labels):
    dict = {"unacc": 1.0, "acc":2.0, "good":3.0, "vgood":4.0}
    for i in range(len(labels)):
        labels[i] = dict[labels[i]]
    return labels