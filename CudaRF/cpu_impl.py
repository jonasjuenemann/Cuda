import random
import operator
from data_cars import cars_1, car_labels_1
from collections import Counter
import pycuda.autoinit
import pycuda.driver as drv
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np

cars = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'],
        ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'],
        ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'],
        ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'],
        ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]
cars = cars_1
# car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']
car_labels = car_labels_1
cars_dict = {0: "Buying Price", 1: "Price of maintenance", 2: "Number of doors", 3: "Person Capacity",
             4: "Size of luggage boot", 5: "Estimated Saftey"}


# Die Arrays waeren zu GPUarrays umwandelbar, man muesste allerdings aufpassen, dass diese dann ihre zweite Dimension verlieren.

def change_data(data):
    dicts = [{'vhigh': 1.0, 'high': 2.0, 'med': 3.0, 'low': 4.0},
             {'vhigh': 1.0, 'high': 2.0, 'med': 3.0, 'low': 4.0},
             {'2': 1.0, '3': 2.0, '4': 3.0, '5more': 4.0},
             {'2': 1.0, '4': 2.0, 'more': 3.0},
             {'small': 1.0, 'med': 2.0, 'big': 3.0},
             {'low': 1.0, 'med': 2.0, 'high': 3.0}]
    for row in data:
        for i in range(len(dicts)):  # len(dicts) = 6 da fuer jedes feature eine Anpassung
            row[i] = dicts[i][row[i]]
    return data


def change_labels(labels):
    dict = {"unacc": 1.0, "acc": 2.0, "good": 3.0, "vgood": 4.0}
    for i in range(len(labels)):
        labels[i] = dict[labels[i]]
    return labels


car_data = change_data(cars)
# car_labels = change_labels(car_labels)

"""Pure Python Variante:"""

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
        """Als Value wird hier das beste Feature zum Teilen gegeben, sehr Praktisch, da man dies in jeder Internal node dann 
        speichert was dazu fuehrt, dass man dieses rauspicken kann."""
        branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


def print_tree(node, question_dict, spacing=""):
    """World's most elegant tree printing function."""
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


tree = build_tree(cars, car_labels)
print_tree(tree, cars_dict)

training_data = car_data[:int(len(car_data) * 0.8)]
training_labels = car_labels[:int(len(car_data) * 0.8)]

testing_data = car_data[int(len(car_data) * 0.8):]
testing_labels = car_labels[int(len(car_data) * 0.8):]

tree = build_tree(training_data, training_labels)
single_tree_correct = 0.0

# test a single tree for accuracy of its predictions for the test set
for i in range(len(testing_data)):
    prediction = classify(testing_data[i], tree)
    # print(prediction)
    # print(testing_labels[i])
    # print(prediction == testing_labels[i])
    if prediction == testing_labels[i]:
        single_tree_correct += 1
print("Percentage of the test_set a single tree predicted accurately")
print(single_tree_correct / len(testing_data))


# Random Forest als Pure Python


def make_random_forest(n, training_data, training_labels):
    trees = []
    for i in range(n):
        indices = [random.randint(0, len(training_data) - 1) for x in range(len(training_data))]

        training_data_subset = [training_data[index] for index in indices]
        training_labels_subset = [training_labels[index] for index in indices]

        tree = build_tree(training_data_subset, training_labels_subset)
        trees.append(tree)
    return trees


forest = make_random_forest(40, training_data, training_labels)
# test a whole forest for accuracy of its predictions for the test set
forest_correct = 0.0
for i in range(len(testing_data)):
    predictions = []
    for forest_tree in forest:
        predictions.append(classify(testing_data[i], forest_tree))
    forest_prediction = max(predictions, key=predictions.count)
    if forest_prediction == testing_labels[i]:
        forest_correct += 1

print("Prediction des forests")
print(forest_correct / len(testing_data))
