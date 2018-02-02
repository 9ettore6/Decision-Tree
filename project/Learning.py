import Fork
import Leaf


def decisionTreeLearner(dataset, pruning=0, m=0):
    target = dataset.target
    values = dataset.values
    examples = dataset.examples
    inputs = dataset.inputs

    class Nodes:  # alternative to global variable
        internal_nodes = 0

    def decision_tree_learning(examples, attrs, m, pruning, parent_examples=()):
        if len(examples) == 0:
            return best_common_value(parent_examples)
        elif all_same_class(examples):
            return Leaf.DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return best_common_value(examples)
        # as parameter I've use the lenght of examples that is decrease at every iteration from removeall
        elif errors(examples, pruning) < m:  # regularization(pre-pruning)
            return best_common_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = Fork.DecisionFork(A, dataset.attrnames[A], best_common_value(examples))
            # create tree with root A, no branches first time
            Nodes.internal_nodes += 1
            for (value, exs) in split(A, examples):
                subtree = decision_tree_learning(exs, removeall(A, attrs), m, pruning, examples)
                tree.add(value, subtree)  # add branch
            return tree

    def errors(examples, pruning):
        if pruning == 0:
            # max number of target value in examples
            counter = 0
            for v in values[target]:
                c = count(target, v, examples)
                if c > counter:
                    counter = c
            return len(examples) - counter
        elif pruning == 1:
            return len(examples)

    def removeall(item, seq):
        """Return a copy of seq with all occurences of item removed."""
        if isinstance(seq, str):
            return seq.replace(item, '')
        else:
            return [x for x in seq if x != item]

    def best_common_value(examples):
        """Return the most popular target value for this set of examples."""
        counter = 0
        popular = 0
        for v in values[target]:
            c = count(target, v, examples)
            if c > counter:
                counter = c
                popular = v
        return Leaf.DecisionLeaf(popular)

    def count(attr, val, examples):
        """Count the number of examples that have attr = val."""
        counter = 0
        for e in examples:
            if e[attr] == val:
                counter += 1
        return counter

    def all_same_class(examples):
        """Are all these examples in the same target class?"""
        _class = examples[0][target]
        for e in examples:
            if e[target] != _class:
                return False
        return True

    def choose_attribute(attrs, examples):
        """Choose the attribute with the highest information gain."""
        best = 0
        for attr in attrs:
            if information_gain(attr, examples) >= best:
                best = information_gain(attr, examples)
                better = attr
        return better  # index(int) of the attribute with best information gain

    def information_gain(attr, examples):
        """Meausure of gain through Gini coefficent, to choose the best attribute"""

        def Gini(examples):
            indexs = 1
            for val in values[target]:
                if len(examples) != 0:
                    prob = float(count(target, val, examples)) / len(examples)
                    indexs -= prob ** 2
            return indexs

        def remainder(examples):
            R = 0
            for (v, examples_k) in split(attr, examples):
                R += (len(examples_k) / (float(len(examples)))) * (Gini(examples_k))
            return R

        return Gini(examples) - remainder(examples)

    def split(attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    tree = decision_tree_learning(examples, inputs, m, pruning)
    return tree, Nodes.internal_nodes
