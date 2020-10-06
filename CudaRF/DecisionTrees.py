# coding=utf-8
import random
from collections import Counter

"""Wir werden den "greedy" approach benutzen, also den größtmoeglichen akutellen gain finden und splitten, unabh. davon ob der split spaeter noch hoeheren Value
haben koennte."""

car = ["med", "med", "4", "4", "big", "high"]
# print(classify(car, tree))
labels = ["unacc", "unacc", "acc", "acc", "good", "good"]
labels1 = ["unacc", "unacc", "unacc", "good", "vgood", "vgood"]
labels2 = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc"]


# Gini Koeffizient zum ermitteln der "purity" ein Wert der angibt, wie eindeutig eine Zuordnung von labels fuer etwas ist. 0 -> sehr gut
def gini(labels):
    impurity = 1
    label_counts = Counter(labels)
    # print(label_counts)
    for label in label_counts:
        probability_of_label = label_counts[label] / len(labels)
        impurity -= probability_of_label ** 2
    return impurity


print("Gini Koeff. von labels")
print(gini(labels))
print(gini(labels1))
print(gini(labels2))

unsplit_labels = ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good", "vgood",
                  "vgood", "vgood"]

split_labels_1 = [
    ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "vgood"],
    ["good", "good"],
    ["vgood", "vgood"]
]

split_labels_2 = [
    ["unacc", "unacc", "unacc", "unacc", "unacc", "unacc", "good", "good", "good", "good"],
    ["vgood", "vgood", "vgood"]
]

"""infomation gain -> macht nur mit weighted gini Koeff. wirklich Sinn, da sonst Randwerte mit geringer Impurity aber sehr niedriger Menge sehr viel Einfluss"""


def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    for subset in split_labels:
        info_gain -= gini(subset) * (
                len(subset) / len(starting_labels))  # macht das ganze zum weighted gini Koeffizienten
    # print(info_gain)
    return info_gain


print("Purity der unsplit labels")
print(gini(unsplit_labels))
print("Information gain von split labels -> tendenziell besser als unsplit labels (die recht unpure sind)")
print(information_gain(unsplit_labels, split_labels_2))

cars = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'],
        ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'],
        ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'],
        ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'],
        ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

"""
Diese Funktion nimmt Data und eine Column entgegen und ermittelt das Subset fuer diese Column sowie die entsprechenden labels fuer
die Eintrage in den Columns
bspw. fuer unsere cars und labels und Column 0 dieser Output.
Hier werden die Eintraege in cars alphabetisch nach dem Wert der Column 0 geordnet und zusammensortiert und dann geguckt, welche labels hier
auftreten.
In unserem Beispiel kann man sehen das bei high und vhigh lediglich unacc Werte (impurity = 0) gibt. (Wobei hier keine große Stichprobe vorliegt)
Genauso bei low nur good Werte -> auch hier wieder: impurity = 0 aber Stichprobe sehr gering.
Bei medium hingegen herrscht sehr große impurity (3versch. Outputs mehrfach) -> macht ja auch Sinn, Randwerte geben eher einen Ausschlag fuer das label als ein mittiger
[[['high', 'med', '3', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low']], [['low', 'low', '2', '4', 'big', 'med']], [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med']], [['vhigh', 'vhigh', '2', '2', 'med', 'low']]]
[['unacc', 'unacc'], ['good'], ['acc', 'acc', 'unacc', 'unacc', 'vgood', 'acc'], ['unacc']]
"""


def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))  # macht eine Liste aus den unique Werten in dataset[column]
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


split_data, split_labels = split(cars, car_labels, 3)
print("Aufspaltung der cars data mit labels nach Column 3")
print(split_data)
print(split_labels)

print("Originale Impurity")
print(gini(car_labels))
print(
    "Information gain (hoch -> gut, bedeutet impurity in den subsets ist tendenziell sehr niedrig) nach Split nach Column 0")
print(information_gain(car_labels, split_labels))
"""Wir gucken durch Einteilung in versch. Untersets um wie viel sich die Impurity "verbessert" -> Alte Impurity - Summe der neuen Impurities"""
"""Machen wir dies Einmal fuer alle columns"""
for i in range(6):
    split_data, split_labels = split(cars, car_labels, i)
    print("Information Gain bei Spaltung nach Column {}".format(i))
    print(information_gain(car_labels, split_labels))

"""Hier waere die entsprechende Funktion, die einmal fuer alle Splits den Information gain nachguckt, und auch direkt den besten zurueckgibt."""


def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


"""Eine der komplizierteren Funktionen in dem ganzen Algorithmus, alles was diese aber letztlich macht ist, die labels nicht wie zuvor als Liste zurueckzuegeben,
sondern als Counter, bspw. {'unacc': 4} anstatt von 4mal unacceptable. Außerdem werden bei verschiedenen Werten im WerteSet diese weiter aufgespalten sodass man dann eine
Liste von Countern enthält. Weiter unten sehen wir auch, dass bei umfangreicheren Datensaetzen auch im Split dann noch weiter aufgesplittet, je nachdem, wie die besten features zum
Splitten in den Unterdatensaetzen liegen.
Was wir nicht sehen, ist fuer welchen Wert (featureWert) hier eig. gesplittet wird, was natuerlich aber eien Info ist, die nachher brauchen, um aus den Ursprungswerten eines neuen Datums
aus sein label schließen zu koennen. Dies wird allerdings in den Klassen Leaf und InternalNode gespeichert"""


def build_tree(data, labels):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain == 0:
        return Counter(labels)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branches.append(build_tree(data_subsets[i], label_subsets[i]))
    return branches


tree = build_tree(cars, car_labels)
print("tree in seiner ersten Form, mit der selbst gebauten Funktion")
print(tree)


def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""
    question_dict = {0: "Buying Price", 1: "Price of maintenance", 2: "Number of doors", 3: "Person Capacity",
                     4: "Size of luggage boot", 5: "Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Counter):
        print(spacing + str(node))
        return

    # Print the question at this node
    print(spacing + "Splitting")

    # Call this function recursively on the true branch
    for i in range(len(node)):
        print(spacing + '--> Branch ' + str(i) + ':')
        print_tree(node[i], spacing + "  ")


print_tree(tree)
print("Reinform des tree (einfach eine Liste von Countern)")
print(tree)

"""
Wuerde auch hier, wenn man zuvor den Datensatz bewusst falsch spaltet, das "richtige" Ergebnis finden
split_data, split_labels = split(cars, car_labels, 1)
for i in range(len(split_data)):
    build_tree(split_data[i], split_labels[i])
    print_tree(tree)
    print(tree)
"""

from data_cars import cars_1, car_labels_1
from tree_w_features import *

"""Hier wird der tree nicht mehr wie bei uns gebildet, sondern mit den Klassen Leaf(value&labels) und Internal_Node(value,branches&features). Entsprechend ist
auch die build_tree Funktion eine andere."""

print("Kleiner Car tree")
tree = build_tree_features(cars, car_labels)
print_tree_features(tree)
print("Reinform des gegebenen build_tree Algorithmus, wie man sieht einfach ein Objekt")
print(tree)

print("Großer Car tree (auskommentiert, zu lang)")
tree = build_tree_features(cars_1, car_labels_1)
#print_tree_features(tree)

"""Teilt das ganze wie der build_tree-Hauptalgorithmus zunaechst in drei Teile, diese sind sozusagen die ersten 3 Branches die der Hauptalgorithmus auch bildet.
Der zweite ist hierbei schon auf 0 impurity, daher werden lediglich die anderen beiden noch weiter aufgespalten"""

# print(find_best_split(cars_1, car_labels_1))  # gibt 5 zurueck
# split_data, split_labels = split(cars_1, car_labels_1, 5)
# print("Aufspaltung der großen cars data mit labels nach Column 5")
# print(split_data)
# print(split_labels)


"""hier darf nur ein tree der ueber build_tree_features() gebaut wurde uebergeben werden, da der "normale" Tree keine labels in seinen Leafs stored (sondern prdictions"""


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


test_point = ['vhigh', 'low', '3', '4', 'med', 'med']
# question_dict = {0: "Buying Price", 1: "Price of maintenance", 2: "Number of doors", 3: "Person Capacity",
# 4: "Size of luggage boot", 5: "Estimated Saftey"}
# braucht nicht uebergeben zu werden, lediglich zum Verstaendnis
print(
    "Klasse eines Testautos ['vhigh', 'low', '3', '4', 'med', 'med'], nach unserem Baum sollte dies \"unacc\" ergeben.")
print(classify(test_point, tree))









"""Um das Ganze jetzt mit sklearn zu machen: Werte muessen angepasst (relativiert) werden:_"""


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


cars_1 = change_data(cars_1)

car_data = cars_1
car_labels = car_labels_1
# man keonnte hier die Daten mit random umshufflen um Zufallspakete von punkten zu erhalten, hierfuer muessten allerdings die labels angehaengt
# und anschließend wieder abhaengt werden.
for i in range(len(car_data)):
    car_data[i].append(car_labels[i])
# print(car_data)
random.shuffle(car_data)
print(car_data)
"""man beachte die Reihenfolge, wenn hier erst car_data ueberschrieben wird, sind die labels weg."""
car_labels = [x[-1] for x in car_data]
car_data = [x[:-1] for x in car_data]
print(car_labels)
# print(car_data)
training_points = car_data[:int(len(car_data) * 0.9)]
training_labels = car_labels[:int(len(car_labels) * 0.9)]
testing_points = car_data[int(len(car_data) * 0.9):]
testing_labels = car_labels[int(len(car_labels) * 0.9):]
print("Erstes Auto im TrainingsSet")
print(training_points[0])
print(training_labels[0])

