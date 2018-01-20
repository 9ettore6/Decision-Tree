from Plot import test, create_dataset

from Learning import decisionTreeLearner


# Car dataset


def car():
    attrnames = ['buying', 'maint', 'doors', 'pearsons', 'lug_boot', 'safety', 'value']  # mnemonic name for attrs
    values = [['vhigh', 'high', 'med', 'low'],
              ['vhigh', 'high', 'med', 'low'],
              ['2', '3', '4', '5more'],
              ['2', '4', 'more'],
              ['small', 'med', 'big'],
              ['low', 'med', 'high'],
              ['unacc', 'acc', 'good', 'vgood']]
    mr = input('Inserire m_range:')
    target = 6
    test('Car.txt', mr, target, 'Car', attrnames, values)
    tree, node = decisionTreeLearner(create_dataset('Car.txt', attrnames, target, values))
    tree.display()


# Tic Tac Toe dataset

def tictactoe():
    attrnames = ['topleft', 'topmiddle', 'topright', 'middleleft', 'middlemiddle', 'middleright', 'bottomleft',
                 'bottommiddle', 'bottomright', 'result']
    values = [['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['x', 'o', 'b'],
              ['positive', 'negative']]
    mr = input('Inserire m_range:')
    target = 9
    test('TicTacToe.txt', mr, target, 'TicTacToe', attrnames, values)
    tree, node = decisionTreeLearner(create_dataset('TicTacToe.txt', attrnames, target, values))
    tree.display()


# Nursery dataset
def nursery():
    attrnames = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'evaluation']
    values = [['usual', 'pretentious', 'great_pret'],
              ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],
              ['complete', 'completed', 'incomplete', 'foster'],
              ['1', '2', '3', 'more'],
              ['convenient', 'less_conv', 'critical'],
              ['convenient', 'inconv'],
              ['nonprob', 'slightly_prob', 'problematic'],
              ['recommended', 'priority', 'not_recom'],
              ['not-recom', 'recommend', 'very_recom', 'priority', 'spec_prior']]
    mr = input('Inserire m_range:')
    target = 8
    test('Nursery.txt', mr, target, 'Nursery', attrnames, values)
    tree, node = decisionTreeLearner(create_dataset('Nursery.txt', attrnames, target, values))
    tree.display()


options = {0: car,
           1: tictactoe,
           2: nursery}
options[2]()
