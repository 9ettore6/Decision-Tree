import Fork
import Leaf


def decisionTreeLearner(dataset, m=0):
    target = dataset.target
    values = dataset.values
    examples = dataset.examples
    inputs = dataset.inputs

    class Nodes:
        internal_nodes = 0

    def decision_tree_learning(examples, attrs, m, parent_examples=()):
        if len(examples) == 0:
            return best_common_value(parent_examples)
        elif all_same_class(examples):
            return Leaf.DecisionLeaf(examples[0][target])
        elif len(attrs) == 0:
            return best_common_value(examples)
        elif errors(examples) < m:  # try to avoid loose of information with a too strict bound
            return best_common_value(examples)
        else:
            A = choose_attribute(attrs, examples)
            tree = Fork.DecisionFork(A, dataset.attrnames[A], best_common_value(examples))
            # create tree with root A, no branches at first
            Nodes.internal_nodes += 1
            for (value, exs) in split(A, examples):
                subtree = decision_tree_learning(exs, removeall(A, attrs), m, examples)
                tree.add(value, subtree)  # add branch
            return tree

    def errors(examples):
        maxval = best_common_value(examples)
        numval = count(target, maxval, examples)  # how many target == maxval?
        return len(examples) - numval

    def removeall(item, seq):
        """Return a copy of seq (or string) with all occurences of item removed."""
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
        return sum(e[attr] == val for e in examples)

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
        for ex in attrs:
            if information_gain(ex, examples) >= best:
                best = information_gain(ex, examples)
                better = ex
        return better  # index(int) of the attribute with best information gain

    def information_gain(attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""

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

    tree = decision_tree_learning(examples, inputs, m)
    return tree, Nodes.internal_nodes
