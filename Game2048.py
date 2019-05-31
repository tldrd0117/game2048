import numpy as np
import random
import copy
from mcts.MCTS import MCTS
from mcts.MCTSNode import MCTSNode
from Game2048Table import TableState
from concurrent.futures import ThreadPoolExecutor
import time
import sys

class GameG2048:
    def __init__(self):
        self.tableState = TableState()
        self.sumReward = 0
        self.maxValue = 2

    def gameInit(self):
        self.sumReward = 0
        self.maxValue = 2
        self.tableState.reset()
        self.tableState.generateNum()
        self.tableState.generateNum()
        return np.array(self.tableState.table).flatten()

    def initGameStart(self):
        self.gameInit()
        while not self.tableState.isEndGame():
            self.tableState.print_table()
            actionStr = input("Action (LEFT: 0, UP: 1, RIGHT: 2, DOWN: 3):")
            action = self.tableState.filterImpossibleAction(int(actionStr))
            if action == -1 :
                print("Impossible Action")
                continue
            print(action)
            reward = self.tableState.step(action)
            print("reward: ", reward)
            self.tableState.generateNum()
    
    def mcts(self):
        self.gameInit()
        mcts = MCTS(self.tableState)
        current_node = MCTSNode(self.tableState)
        
        while not current_node.state.isEndGame():
            print('before')
            # current_node = MCTSNode(current_node.state)
            current_node.state.print_table()
            # nodes = [[25,copy.deepcopy(current_node)] for _ in range(4)]
            nodes = [copy.deepcopy(current_node) for _ in range(5)]
            finals = []
            beforeTime = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                for value in executor.map(mcts.UCTSEARCH_FULL, nodes):
                    action = -1
                    index = 0
                    while action == -1:
                        action = current_node.state.filterImpossibleAction(int(value[index]['action']))
                        if action == -1 :
                            index+=1
                            print("Impossible Action")
                            continue
                    finals.append(value[index])
            print('time: ', time.time() - beforeTime)
            nodeAvg = max(finals, key=lambda item: item['avg'])
            action = nodeAvg['action']
            # nodeAvg = mcts.UCTSEARCH(10, current_node)
            # print('action')
            # action = -1
            # index = 0
            # while action == -1:
            #     action = current_node.state.filterImpossibleAction(int(nodeAvg[index]['action']))
            #     if action == -1 :
            #         index+=1
            #         print("Impossible Action")
            #         continue
            print('action: ', action)
            reward = current_node.state.step(action)
            print("reward: ", reward)
            current_node.state.generateNum()
            for child in nodeAvg['child']:
                allTrue = True
                for i in range(0,4):
                    for j in range(0,4):
                        allTrue = child.state.table[i][j] == current_node.state.table[i][j]
                if allTrue:
                    current_node = child
                    break


        print('ENDGAME')
        current_node.state.print_table()

    def step(self, action):
        done = False
        if self.tableState.isEndGame():
            done = True
        a = self.tableState.filterImpossibleAction(action)
        if a == -1:
            done = True

        reward = self.tableState.step(a)
        self.tableState.generateNum()
        self.tableState.print_table()
        self.maxValue = np.array(self.tableState.table).max()

        return np.array(self.tableState.table).flatten(), reward, done, 0

if __name__ == "__main__":
    game = GameG2048()
    # game.simulateGameStart()
    sys.setrecursionlimit(100000)
    game.mcts()
    


