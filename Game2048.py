import numpy as np
import random
import copy
from mcts.MCTS import MCTS
from mcts.MCTSNode import MCTSNode
from Game2048Table import TableState
from concurrent.futures import ThreadPoolExecutor
import time
import sys
from mcts.record.Recorder import Recorder
import signal
from collections import deque

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
        step = 0
        record = []
        rewards = [0]
        actions = [0]
        while not current_node.state.isEndGame():
            print('before')
            # current_node = MCTSNode(current_node.state)
            current_node.state.print_table()
            record.append({'table': current_node.state.table, 'reward': rewards[step], 'step': step, 'action': actions[step]})
            # nodes = [[25,copy.deepcopy(current_node)] for _ in range(4)]
            nodes = [copy.deepcopy(current_node) for _ in range(1)]
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
            rewards.append(reward)
            actions.append(action)
            print("reward: ", reward)
            step += 1
            print('step: ', step)
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
        return np.sum(current_node.state.table), record
    
    def mcts_policy(self, POLICY):
        self.gameInit()
        mcts = MCTS(self.tableState)
        current_node = MCTSNode(self.tableState)
        step = 0
        record = deque()
        rewards = [0]
        actions = [0]
        score = 0
        while not current_node.state.isEndGame():
            print('before')
            # current_node = MCTSNode(current_node.state)
            current_node.state.print_table()
            # nodes = [[25,copy.deepcopy(current_node)] for _ in range(4)]
            node = copy.deepcopy(current_node)
            finals = []
            beforeTime = time.time()
            value = mcts.UCTSEARCH_POLICY(node, POLICY)
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
            history = np.array(current_node.state.table).flatten()
            reward = current_node.state.step(action)
            next_history = np.array(current_node.state.table).flatten()
            dead = current_node.state.isEndGame()
            rewards.append(reward)
            actions.append(action)
            print("reward: ", reward)
            step += 1
            score += reward
            print('step: ', step)
            current_node.state.generateNum()
            record.append((history, action, reward, next_history, dead))

            
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
        return np.sum(current_node.state.table), record, score, step

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
    
    def mctsRecord(self):
        recorder = Recorder()
        recorder.create()
        testValues = [0.0, 0.3, 0.6, 0.9]

        for value1 in range(0,4):
            for value2 in range(0,4):
                for value3 in range(0,4):
                    TableState.param1 = testValues[value1]
                    TableState.param2 = testValues[value2]
                    TableState.param3 = testValues[value3]
                    reward, values = game.mcts()
                    tables = values[0]['table']
                    tableNum = [0] * 16
                    for table in tables:
                        for idx, space in enumerate(table):
                            if space > 0:
                                tableNum[idx]+=1
                    print(tableNum)
                    result = { "value": str(reward), 'param':[str(TableState.param1), str(TableState.param2), str(TableState.param3)], 'tableNum': tableNum}
                    recorder.save(result)
    def mctsRecordTable(self):
        recorder = Recorder()
        recorder.create()
        _, record = self.mcts()
        record[0]
        
def signal_handler(signal, frame):
    print(TableState.param1,TableState.param2,TableState.param3)
    sys.exit(0)


if __name__ == "__main__":
    game = GameG2048()
    game.initGameStart()
    # sys.setrecursionlimit(100000)
    # signal.signal(signal.SIGINT, signal_handler)
    # TableState.param1 = 0.4
    # TableState.param2 = 0.7
    # TableState.param3 = 0.2
    # game.mctsRecord()


