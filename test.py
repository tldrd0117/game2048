import unittest
import copy
from Game2048Table import TableState
from mcts.MCTSNode import MCTSNode
class Tests(unittest.TestCase):
    def setUp(self):
        self.state = TableState()
        self.state.table[0][0] = 2
        self.state.table[0][1] = 2
        print('setup')
    def test_runs(self):
        node = MCTSNode(self.state)
        # copyState = copy.deepcopy(self.state)
        # copyState.action = 0
        # reward = copyState.step(0)
        # copyState.table[0][1] = 2
        # node.add_child(copyState, reward)
        print('1')
        # copyState.print_table()
        node.state.print_table()
        newstate, reward = node.state.createPossibleRandomCaseChildState([c.state for c in node.children])
        newstate.print_table()
        
        print('2')
        node.state.print_table()
        node.add_child(newstate, reward)
        newstate, reward = node.state.createPossibleRandomCaseChildState([c.state for c in node.children])
        print('3')
        newstate.print_table()
        node.state.print_table()
        node.add_child(newstate, reward)
        newstate, reward = node.state.createPossibleRandomCaseChildState([c.state for c in node.children])
        print('4')
        newstate.print_table()
        node.state.print_table()
        node.add_child(newstate, reward)
        print('test_runs')
    def tearDown(self):
        print('teardown')
if __name__ == "__main__":
    unittest.main()