import random
import operator
from collections import Counter

random.seed(1)


def unique_labels(rows, feature):
    labels = []
    for row in rows:
        if row[feature] not in labels:
            labels.append(row[feature])
    return labels


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / float(len(rows))
        impurity -= prob_of_label ** 2
    return impurity


def partition(rows, feature, value):
    if is_numeric(rows[0][feature]):
        true_rows, false_rows = [], []
        for row in rows:
            if row[feature] >= value:
                true_rows.append(row)
            else:
                false_rows.append(row)
        return [true_rows, false_rows]
    else:
        groups = []
        counts = unique_labels(rows, feature)
        for k in counts:
            new_list = []
            for row in rows:
                if row[feature] == k:
                    new_list.append(row)
            groups.append(new_list)
        return groups


def partition_word(rows, feature):
    groups = []
    counts = unique_labels(rows, feature)
    for k in counts:
        new_list = []
        for row in rows:
            if row[feature] == k:
                new_list.append(row)
        groups.append(new_list)
    return groups


def info_gain(children, current_uncertainty):
    gain = current_uncertainty
    total = 0
    for child in children:
        total += len(child)
    for child in children:
        p = float(len(child)) / total
        gain -= p * gini(child)
    return gain


def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            # try splitting the dataset
            splits = partition_word(rows, col)

            # Skip this split if it doesn't divide the
            # dataset.
            if len(splits) == 1:
                continue

            # Calculate the information gain from this split
            gain = info_gain(splits, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, [col, val]

    return best_gain, best_question


class Leaf:

    def __init__(self, rows, value):
        self.predictions = class_counts(rows)
        self.value = value


class Decision_Node:

    def __init__(self,
                 question,
                 branches, value):
        self.question = question
        self.branches = branches
        self.value = value


def build_tree(rows, value):
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.
    result = find_best_split(rows)
    gain = result[0]
    question = result[1]

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows, value)

    # If we reach here, we have found a useful feature / value
    # to partition on.
    children = partition_word(rows, question[0])
    branches = []
    for child in children:
        branch = build_tree(child, child[0][question[0]])
        branches.append(branch)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # dependingo on the answer.
    return Decision_Node(question, branches, value)


def print_tree_prediction(node, question_dict=None, spacing=""):
    """World's most elegant tree printing function."""
    if question_dict is None:
        question_dict = {0: "Buying Price", 1: "Price of maintenance", 2: "Number of doors", 3: "Person Capacity",
                         4: "Size of luggage boot", 5: "Estimated Saftey"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print(spacing + question_dict[node.question[0]])

    # Call this function recursively on the true branch
    for i in range(len(node.branches)):
        print(spacing + '--> Branch ' + node.branches[i].value + ':')
        print_tree_prediction(node.branches[i], question_dict, spacing + "  ")


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


def classify(row, node):
    if isinstance(node, Leaf):
        return max(node.predictions.items(), key=operator.itemgetter(1))[0]
    answer = row[node.question[0]]
    for branch in node.branches:
        if branch.value == answer:
            return classify(row, branch)


def make_cars():
    f = open("data_car.data", "r")
    cars = []
    for line in f:
        cars.append(line.rstrip().split(","))
    return cars


#data = make_cars()
#tree = build_tree(data, "")
