import random
import pydotplus
from IPython.core.display import Image
from copy import deepcopy
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.externals.six import StringIO

from Dataset import DataSet
from Learning import decisionTreeLearner

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Image(graph.write_png('tree.png'))


attrnames = [['buying'], ['maint'], ['doors'], ['pearsons'], ['lug_boot'], ['safety'], ['value']]
values = [['v-high', 'high', 'med', 'low'], ['v-high', 'high', 'med', 'low'], ['2', '3', '4', '5more'],
          ['2', '4', 'more'], ['small', 'med', 'big'],
          ['low', 'med', 'high'], ['unacc', 'acc', 'good', 'v-good']]


def test():
    data = create_dataset('Car.txt', attrnames, values[6].index('unacc'), values)
    sets = createSet(data)
    trainset = sets[0]
    testset = sets[1]
    dot_data = StringIO()
    tre, node = decisionTreeLearner(data, 20)
    tre.display()


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


test()
