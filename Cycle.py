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
from DeepLearning import DQNAgent
import os
import signal
import logging
import pickle

logging.basicConfig(handlers=[logging.FileHandler('log/simulation1.log', 'a', 'utf-8')], level=logging.INFO, format='%(message)s')
def printG(*msg):
    joint = ' '.join(list(map(lambda x : str(x), msg)))
    print(joint)
    logging.info(joint)

printG('')

def printGTable(table):
    printG("------------------------")
    for i in range(4):
        printG("%5d %5d %5d %5d"%(table[i][0], table[i][1], table[i][2], table[i][3]))
    printG("------------------------")

class Cycle:
    global_step = 0
    def POLICY(self, state):
        copyState = copy.deepcopy(state)
        reward = 0
        while copyState.isEndGame()==False:
            action = copyState.getPossibleRandomAction()
            reward += copyState.step(action)
            copyState.generateNum()
        return reward
    def cycle_(self, game):
        self.agent = DQNAgent(action_size=4)
        self.predictSum = [0,0,0,0]
        e = 0
        targetEpisode = 50000
        for _ in range(targetEpisode):
            e += 1
            self.recentPredictSum = [0,0,0,0]
            agent = self.agent
            if _ % 3 == 0:
                printG('predict policy')
                lastTable, record, score, step = game.mcts_policy(cycle.DQN_POLICY_PREDICT)
            else:
                printG('random + predict policy')
                lastTable, record, score, step = game.mcts_policy(cycle.DQN_POLICY)
            agent.appends_sample(record)
            self.global_step+=step
            printG('predictSum',self.predictSum)
            printG('recentPredictSum', self.recentPredictSum)
            printG('step',step)
            printG('max_number',np.max(np.array(lastTable)))
            printG('table_number_sum', np.sum(np.array(lastTable)))
            printGTable(lastTable)
            printG(len(agent.memory), self.global_step, agent.update_target_rate, agent.train_start)
            if len(agent.memory) >= agent.train_start:
                for _ in range(10):
                    train_loss = agent.train_model()
                    printG('loss', train_loss)
            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if self.global_step % agent.update_target_rate == 0:
                agent.update_target_model()
            
            if self.global_step > agent.train_start:
                stats = [score, agent.avg_q_max / float(step), step,
                            agent.avg_loss / float(step)]
                for i in range(len(stats)):
                    agent.sess.run(agent.update_ops[i], feed_dict={
                        agent.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = agent.sess.run(agent.summary_op)
                agent.summary_writer.add_summary(summary_str, e + 1)

                printG("episode:", e, "  score:", score, "  memory length:",
                        len(agent.memory), "  epsilon:", agent.epsilon,
                        "  global_step:", self.global_step, "  average_q:",
                        agent.avg_q_max / float(step), "  average loss:",
                        agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0
            # if e % 10 == 0:
            agent.model.save_weights("./save_model/game2048_dqn.h5")
            agent.saveDeque()
    
    def DQN_POLICY(self, state):
        reward = 0
        step = 0
        
        copyState = copy.deepcopy(state)
        while copyState.isEndGame()==False:
            step+=1
            # self.global_step +=1
            history = np.array([[y for x in copyState.table for y in x]])
            history = history.reshape(1,1,4,4)
            actions, isPredicted = self.agent.get_action(history)
            if isPredicted:
                maxValue = np.max(actions)
                self.predictSum[np.where(maxValue == actions)[0][0]] += 1
                self.recentPredictSum[np.where(maxValue == actions)[0][0]] += 1
            action = copyState.maxPossibleAction(actions)

            if action != -1:
                reward += copyState.step(action)
                copyState.generateNum()
            else:
                print('impossible')
        # print(step)
        return reward
        

    def DQN_POLICY_PREDICT(self, state):
        reward = 0
        step = 0
        
        copyState = copy.deepcopy(state)
        while copyState.isEndGame()==False:
            step+=1
            # self.global_step +=1
            history = np.array([[y for x in copyState.table for y in x]])
            history = history.reshape(1,1,4,4)
            actions, isPredicted = self.agent.get_action_predict(history)
            if isPredicted:
                maxValue = np.max(actions)
                self.predictSum[np.where(maxValue == actions)[0][0]] += 1
                self.recentPredictSum[np.where(maxValue == actions)[0][0]] += 1
            action = copyState.maxPossibleAction(actions)

            if action != -1:
                reward += copyState.step(action)
                copyState.generateNum()
            else:
                print('impossible')
        # print(step)
        return reward
    def signal_handler(self, signal, frame):
        self.agent.model.save_weights("./save_model/game2048_dqn.h5")
        self.agent.saveDeque()
        sys.exit(0)
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    game = GameG2048()
    cycle = Cycle()
    signal.signal(signal.SIGINT, cycle.signal_handler)
    sys.setrecursionlimit(100000)
    cycle.cycle_(game)
    # _, record = game.mcts_policy(cycle.POLICY)