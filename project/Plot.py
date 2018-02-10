from __future__ import print_function
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from Dataset import DataSet
from Learning import decisionTreeLearner


# iterate several times to avoid that favour/unfavour example compromise the test
def test(file, mr, target, name, attrnames, values, pruning):
    print(' ')
    print('Learning', end='', )
    trainerr = []
    test_err = []
    internal_nodes = []
    for i in range(10):
        j = 0  # from 0 to mr
        for m in range(mr, 0, -1):
            data = create_dataset(file, attrnames, target, values)
            data.examples, tes = createSets(data)
            tree, nodes = decisionTreeLearner(data, pruning, m)
            if i == 0:
                trainerr.append(count_errors(data.examples, target, tree))
                test_err.append(count_errors(tes, target, tree))
                internal_nodes.append(nodes)
            elif i == 9:  # last iteration: average on all results
                trainerr[j] = trainerr[j] / 10
                test_err[j] = test_err[j] / 10
                internal_nodes[j] = internal_nodes[j] / 10
            else:
                trainerr[j] += count_errors(data.examples, target, tree)
                test_err[j] += count_errors(tes, target, tree)
                internal_nodes[j] += nodes
            j += 1
        print('.', end='')
    print('\n')
    plt.title(name)
    plt.plot(internal_nodes, trainerr, label="Training set")
    plt.plot(internal_nodes, test_err, label="Test set")
    plt.legend()
    plt.ylabel('Percentage error')
    plt.xlabel('Internal nodes')
    plt.show()


# count the number of error, that is if predict value is different from target
def count_errors(examples, target, tree):
    counter = 0
    for ex in examples:
        desired = ex[target]
        predicted = tree(ex)
        if desired != predicted:
            counter += 1
    return float(counter) / len(examples) * float(100)


# hold-out cross validation
def createSets(dataset):
    random.shuffle(dataset.examples)
    bound = int((len(dataset.examples) / 100) * 80)
    train = dataset.examples[0:bound]
    test = dataset.examples[bound:len(dataset.examples)]  # bound not included
    return train, test


# format dates to send to Dataset class
def create_dataset(file, attrnames, target, values):
    examples = formatfile(file)
    attrs = [k for k in range(len(examples[0]))]  # index of examples
    inputs = create_input(attrs, target)
    return DataSet(file, examples, inputs, attrs, target, attrnames, values)


def formatfile(file):
    data = []
    f = open(file)
    for line in f.readlines():
        row = line.split(',')  # transform line in list
        row = [r.rstrip() for r in row]  # format lines deleting unuseful infos as "/n"
        data.append(row)  # data: list that contains list of example
    return data


# Returns a list of attributes without the target
def create_input(attributes, target):
    inputs = deepcopy(attributes)
    del inputs[target]  # remove target element(index of target)
    return inputs
