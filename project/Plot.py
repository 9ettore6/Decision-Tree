# This Python file uses the following encoding: utf-8
import os, sys
import random
import math
from copy import deepcopy

import matplotlib.pyplot as plt
from Dataset import DataSet
from Learning import decisionTreeLearner

'''iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Image(graph.write_png('tree.png'))
'''

attrnames = ['buying', 'maint', 'doors', 'pearsons', 'lug_boot', 'safety', 'value']  # mnemonic name for attrs
values = [['vhigh', 'high', 'med', 'low'],
          ['vhigh', 'high', 'med', 'low'],
          ['2', '3', '4', '5more'],
          ['2', '4', 'more'],
          ['small', 'med', 'big'],
          ['low', 'med', 'high'],
          ['unacc', 'acc', 'good', 'vgood']]


# iterate several times to avoid that favour/unfavour example compromise my test
def test(mr, target):
    trainerr = []
    test_err = []
    internal_nodes = []
    for i in range(10):
        j = 0
        for m in range(mr, 0, -1):
            data = create_dataset('Car.txt', attrnames, target, values)
            data.examples, tes = createSets(data)
            tree, node = decisionTreeLearner(data, m)
            if i == 0:
                trainerr.append(count_errors(data.examples, target, tree))
                test_err.append(count_errors(tes, target, tree))
                internal_nodes.append(node)
            elif i == 9:
                trainerr[j] = float("%.3f" % (trainerr[j] / 10))
                print trainerr
                test_err[j] = float("%.3f" % (test_err[j] / 10))
                internal_nodes[j] = math.floor(internal_nodes[j] / 10)
            else:
                trainerr[j] += float("%.3f" % count_errors(data.examples, target, tree))
                test_err[j] += float("%.3f" % count_errors(tes, target, tree))
                internal_nodes[j] += node
            j += 1
    plt.plot(internal_nodes, trainerr, label="Training set")
    plt.plot(internal_nodes, test_err, label="Test set")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('Percentage error')
    plt.xlabel('Internal nodes')
    plt.show()


def count_errors(examples, target, tree):
    counter = 0
    for ex in examples:
        desired = ex[target]
        predicted = tree(ex)
        if desired != predicted:
            counter += 1
    return float(counter / len(examples)) * float(100)


def createSets(dataset):
    random.shuffle(dataset.examples)
    # print(examples) ok
    bound = int((len(dataset.examples) / 100) * 80)
    train = dataset.examples[0:bound]
    test = dataset.examples[bound:len(dataset.examples)]  # bound not included
    return train, test


def create_dataset(file, attrnames, target, values):  # format dates to send its at Dataset class
    """Create a dataset from a file, given its attrs names, values and index target"""
    examples = formatfile(file)
    # examples = create_examples(data)
    attrs = [k for k in range(len(examples[0]))]  # creo gli interi per indicizzare gli esempi
    inputs = create_input(attrs, target)
    return DataSet(file, examples, inputs, attrs, target, attrnames, values)


def formatfile(file):
    data = []
    f = open(file)
    for line in f.readlines():
        row = line.split(',')  # transform line in list
        row = [r.rstrip() for r in row]  # format lines deleting unuseful infos "/n"
        data.append(row)  # data: list that contains list of example
    return data


def create_examples(data):  # INUTILE????
    examples = []
    for i in range(len(data)):
        example = []
        for j in range(len(data[0])):
            example.append(data[i][j])  # creo liste di esempi poi metto tutto in examples(lista di liste)
        examples.append(example)
    return examples


def create_input(attributes, target):
    """Returns a list of attributes without the target"""
    inputs = deepcopy(attributes)
    del inputs[target]  # remove target element(index of target)
    return inputs


target = 6
test(5, target)
tre, node = decisionTreeLearner(create_dataset('Car.txt', attrnames, target, values))
# tre.display()
