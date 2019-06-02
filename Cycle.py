import tensorflow as tf
import numpy as np
from mcts.record.Recorder import Recorder

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras import backend
from Game2048 import GameG2048
import sys
import copy
 
class Cycle:
    def POLICY(self, state):
        copyState = copy.deepcopy(state)
        reward = 0
        while copyState.isEndGame()==False:
            action = copyState.getPossibleRandomAction()
            reward += copyState.step(action)
            copyState.generateNum()
    def cycle(self, game):
        pass
    def train(self):
        pass
    def save(self):
        pass
    def load(self):
        pass

if __name__ == "__main__":
    game = GameG2048()
    cycle = Cycle()
    sys.setrecursionlimit(100000)
    _, record = game.mcts_policy(cycle.POLICY)
    print(record)