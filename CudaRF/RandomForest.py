# coding=utf-8
"""General Idea is to create Multiple Decision Trees on the same (features and labels) with different Pruning and Parameters.
This will lead to overfitting having less of an impact."""

from tree import *
from tree_w_features import *
from data_cars import cars_1, car_labels_1, change_data
import random
import numpy as np

random.seed(4)

tree = build_tree(cars_1, car_labels_1)
# print_tree_prediction(tree)
car_data = cars_1
car_labels = car_labels_1
"""Unser Baum ist leicht anders als bisher (und auf CC) da hier eine Prediction gegeben wird, basierend darauf welche Werte in den einzelnen Leafs sind (bisher einfach immer das Max genommen).
Fuer das Aufsetzen von Forests aus Baumdiagrammen spielt das aber keine Rolle.
Wir wenden hier "bagging an, bedeutet wir erstellen kleinere Subsets unseres Datensatzes (chosen at random with replacement, ein einzelner Datensatz kann also mehrfach vorkommen."""
indices = [random.randint(0, 999) for x in range(0, 1000)]
data_subset = [car_data[i] for i in indices[:100]]
labels_subset = []
for i in indices[:100]:
    labels_subset.append(car_labels[i])
#print(data_subset)
#print(labels_subset)
#print(len(data_subset))
subset_tree = build_tree(data_subset, labels_subset)
# print_tree_prediction(subset_tree)

unlabeled_point = ['high', 'vhigh', '3', 'more', 'med', 'med']
predictions = []

"""Hier wenden wir normales Bagging an, wir erstellen 20 Baeume aus den 1000 Werten, diese wurden vertauscht und eventl. manche mehrfach genommen (andere ausgelassen).
Best practice ist hier, die Gesamtgroeße des Datensets zu nehmen (also n)"""

for i in range(20):
    indices = [random.randint(0, 999) for i in range(1000)]
    data_subset = [car_data[index] for index in indices]
    labels_subset = [car_labels[index] for index in indices]
    subset_tree = build_tree(data_subset, labels_subset)
    # print(classify(unlabeled_point, subset_tree))
    predictions.append(classify(unlabeled_point, subset_tree))
print("lists of predictions when running the same algorithm 20 times")
print(predictions)
final_prediction = max(predictions, key=predictions.count)
print("final prediction:")
print(final_prediction)
# car_data = change_data(car_data)

car_data = change_data(cars_1)
#car_labels = change_labels(car_labels)
training_data = car_data[:int(len(car_data) * 0.8)]
training_labels = car_labels[:int(len(car_data) * 0.8)]

testing_data = car_data[int(len(car_data) * 0.8):]
testing_labels = car_labels[int(len(car_data) * 0.8):]

"""Ab hier muessen wir wir wieder mit der Datenstruktur aus tree_w_features anstatt mit den Predictions arbeiten. (ansonsten erkennt unsere for-SChleife die Prediction nicht ordentlich)"""

tree = build_tree_features(training_data, training_labels)
single_tree_correct = 0.0

#print(classify_feature(testing_data[0], tree))
"""test a single tree for accuracy of its predictions for the test set"""
for i in range(len(testing_data)):
    prediction = classify_feature(testing_data[i], tree)
    #print(prediction)
    #print(testing_labels[i])
    #print(prediction == testing_labels[i])
    if prediction == testing_labels[i]:
        single_tree_correct += 1
        #print(single_tree_correct)
print("Percentage of the test_set a single tree predicted accurately")
print(single_tree_correct / len(testing_data))

"""n -> number of tress in forest"""


def make_random_forest(n, training_data, training_labels):
    trees = []
    for i in range(n):
        indices = [random.randint(0, len(training_data) - 1) for x in range(len(training_data))]

        training_data_subset = [training_data[index] for index in indices]
        training_labels_subset = [training_labels[index] for index in indices]

        tree = build_tree_forest(training_data_subset, training_labels_subset)
        trees.append(tree)
    return trees


forest = make_random_forest(40, training_data, training_labels)
# print(forest) macht keinen Sinn, lediglich gespeicherte Objekte, 40 beaeume komplett auszuprinten dauert allerdings vieeeeeeel zu lang.
"""test a whole for accuracy of its predictions for the test set"""
forest_correct = 0.0
for i in range(len(testing_data)):
    predictions = []
    for forest_tree in forest:
        predictions.append(classify_feature(testing_data[i], forest_tree))
    forest_prediction = max(predictions, key=predictions.count)
    if forest_prediction == testing_labels[i]:
        forest_correct += 1

print("Prediction des forests")
print(forest_correct / len(testing_data))

test_point = ['vhigh', 'low', '3', '4', 'med', 'med']
print(
    "Klasse eines Testautos ['vhigh', 'low', '3', '4', 'med', 'med'], nach unserem Baum sollte dies \"unacc\" ergeben.")
print(classify_feature(test_point, tree))
