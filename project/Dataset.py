class DataSet:
    """A data set for a machine learning problem. It has the following fields:
    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attrnames  Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.setproblem.
                 If not None, an erroneous value raises ValueError.
    d.name       Name of the data set (for output display only)."""

    def __init__(self, name='', examples=None, inputs=None, attrs=None,
                 target=None, attrnames=None, values=None):
        self.name = name
        self.examples = examples
        self.target = target
        self.values = values
        self.inputs = inputs
        # Attrs are the indices of examples, unless otherwise stated.
        if attrs is None and self.examples is not None:
            attrs = list(range(len(self.examples[0])))
        self.attrs = attrs
        # Initialize .attrnames from string, list, or by default
        if isinstance(attrnames, str):
            self.attrnames = attrnames.split()
        else:
            self.attrnames = attrnames or attrs

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Check validity of each values of examples"""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Value: ' + example[a] + ' not valid for ' + self.attrnames[a])

    def remove_examples(self, i):
        """Remove examples that contain given value."""
        if len(self.examples) > i:
            raise IndexError('Index not valid!')
