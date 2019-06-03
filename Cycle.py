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
        e = 0
        targetEpisode = 50000
        for _ in range(targetEpisode):
            e += 1
            agent = self.agent
            _, record, score, step = game.mcts_policy(cycle.DQN_POLICY)
            agent.appends_sample(record)
            self.global_step+=step
            print(len(agent.memory), self.global_step, agent.update_target_rate, agent.train_start)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
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

                print("episode:", e, "  score:", score, "  memory length:",
                        len(agent.memory), "  epsilon:", agent.epsilon,
                        "  global_step:", self.global_step, "  average_q:",
                        agent.avg_q_max / float(step), "  average loss:",
                        agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0
            # if e % 10 == 0:
            agent.model.save_weights("./save_model/game2048_dqn.h5")
        
    def DQN_POLICY(self, state):
        reward = 0
        step = 0
        copyState = copy.deepcopy(state)
        while copyState.isEndGame()==False:
            step+=1
            # self.global_step +=1
            history = np.array(copyState.table).flatten()
            actions = self.agent.get_action(history)
            action = state.maxPossibleAction(actions)
            if action != -1:
                reward += copyState.step(action)
                copyState.generateNum()
        return reward

if __name__ == "__main__":
    game = GameG2048()
    cycle = Cycle()
    sys.setrecursionlimit(100000)
    cycle.cycle_(game)
    # _, record = game.mcts_policy(cycle.POLICY)