import random

import math
import pydotplus
from IPython.core.display import Image
from copy import deepcopy
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO
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

attrnames = ['buying', 'maint', 'doors', 'pearsons', 'lug_boot', 'safety', 'value']
values = [['v-high', 'high', 'med', 'low'],
          ['v-high', 'high', 'med', 'low'],
          ['2', '3', '4', '5more'],
          ['2', '4', 'more'],
          ['small', 'med', 'big'],
          ['low', 'med', 'high'],
          ['unacc', 'acc', 'good', 'v-good']]


def test(mr, target):
    trainerr = []
    test_err = []
    internal_nodes = []
    for i in range(10):
        j = 0
        for m in range(mr, 0, -1):
            data = create_dataset('Car.txt', attrnames, target, values)
            data.examples, tes = createSet(data)
            tre, node = decisionTreeLearner(data, 20)
            if i == 0:
                trainerr.append(count_errors(data.examples, target, tre))
                test_err.append(count_errors(tes, target, tre))
                internal_nodes.append(node)
            elif i == 9:
                trainerr[j] = float("%.3f" % (trainerr[j] / 10))
                test_err[j] = float("%.3f" % (test_err[j] / 10))
                internal_nodes[j] = math.floor(internal_nodes[j] / 10)
            else:
                trainerr[j] += float("%.3f" % count_errors(data.examples, target, tre))
                test_err[j] += float("%.3f" % count_errors(tes, target, tre))
                internal_nodes[j] += node
            j += 1
    plt.plot(internal_nodes, trainerr, label="Training set")
    plt.plot(internal_nodes, test_err, label="Test set")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('Percentage error')
    plt.xlabel('Internal nodes')
    plt.show()


def count_errors(examples, target, tree):
    i = 0
    for e in examples:
        if e[target] != tree(e):
            i += 1
    return i / len(examples) * 100


def createInput(attributes, target):
    """Returns a list of attributes without the target"""
    inputs = deepcopy(attributes)
    del inputs[target]  # rimuove l'elemento di indice target
    return inputs


def createSet(dataset):
    random.shuffle(dataset.examples)
    # print(examples) ok
    bound = int((len(dataset.examples) / 100) * 0.8)
    train = dataset.examples[0:bound]
    test = dataset.examples[bound:len(dataset.examples)]
    return train, test


def create_dataset(file, attrnames, target, values):
    """Create a dataset from a file, given its attrs names, values and index target"""
    data = []
    parts = []
    for line in open(file).readlines():
        parts = line.rstrip()
        parts = line.split(',')
        parts = [p.rstrip() for p in parts]  # formatto line eliminando info superflue "\n"
        data.append(parts)
    examples = []
    for i in range(len(data)):
        example = {}
        for j in range(len(data[0])):
            example[j] = data[i][j]  # creo liste di esempi e poi metto tutto in examples una liste di liste di esempi
        examples.append(example)
    attrs = [k for k in range(len(examples[0]))]  # creo gli interi per indicizzare gli esempi
    inputs = createInput(attrs, target)

    return DataSet(file, examples, inputs, attrs, target, attrnames, values)


target = 1
test(10, target)
tre, node = decisionTreeLearner(create_dataset('Car.txt', attrnames, target, values))
tre.display()
