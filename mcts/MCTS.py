import math
import random
import logging
import copy
from concurrent.futures import ProcessPoolExecutor
class MCTS:
    DEFAULT_SCALAR = 1/math.sqrt(2.0)
    # DEFAULT_SCALAR = 1/math.sqrt(1.0)
    def __init__(self, state):
        self.state = state
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger('MyLogger')
    def search(self, root):
        front=self.TREEPOLICY(root)
        reward=self.DEFAULTPOLICY(front.state)
        self.BACKUP(front,reward)
    def UCTSEARCH_PARAM(self, param):
        budget = param[0]
        root = param[1]
        for _ in range(budget):
            self.search(root)
        result = self.RANKCHILD(root,0)
        # print('resultTable')
        # print(result[0]['child'][0].state.table)
        # print(result[0]['avg'])
        return result
    def UCTSEARCH_FULL(self, root):
        budget = root.state.move_case()
        for _ in range(budget):
            self.search(root)
        return self.RANKCHILD(root,0)


    def UCTSEARCH(self, budget, root):

        for _ in range(budget):
            self.search(root)
            # front=self.TREEPOLICY(root)
            # reward=self.DEFAULTPOLICY(front.state)
            # self.BACKUP(front,reward)
            

        # print(root.state.table)
        # print([ c.state.table for c in root.children])
        result = self.RANKCHILD(root,0)
        # print('resultTable')
        # print(result[0]['child'][0].state.table)
        # print(result[0]['avg'])
        return result
    def TREEPOLICY(self, node):
        tried_children=[c.state for c in node.children]
        # 현재 노드의 자식과 중복되지 않은 새로운 상태를 만든다.
        while node.fully_expanded() == False:
            tried_children=[c.state for c in node.children]
            new_state, reward = node.state.createPossibleRandomCaseChildState(tried_children)
            node.add_child(new_state, reward)
        #자식을 반환한다
        return self.BESTCHILD(node)
        # self.EXPAND(node)
        # while node.state.isEndGame()==False:
        #     #자식노드가 없으면 추가한다
        #     if len(node.children)==0:
        #         return self.EXPAND(node)
        #     #탐험
        #     elif random.uniform(0,1)<.5:
        #         node=self.BESTCHILD(node)
        #     else:
        #         # 자식노드가 꽉차지 않으면 추가한다
        #         if node.fully_expanded()==False:
        #             return self.EXPAND(node)
        #         # 자식노드가 꽉차면 베스트 자식을 선정한다
        #         else:
        #             node=self.BESTCHILD(node)
        # 베스트 자식을 내보낸다
        return node
    def BESTCHILD(self, node, scalar = DEFAULT_SCALAR):
        # bestscore=0.0
        # bestchildren=[]
        #자식에게 점수를 매겨 가장 최고점 자식을 골라낸다
        # for c in node.children:
        #     exploit=c.reward/c.visits
        #     explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))
        #     score=exploit+scalar*explore
        #     if score==bestscore:
        #         bestchildren.append(c)
        #     if score>bestscore:
        #         bestchildren=[c]
        #         bestscore=score
        # if len(bestchildren)==0:
        #     self.logger.warn("OOPS: no best child found, probably fatal")
        #최고점 자식중 베스트를 구한다
        # return random.choice(bestchildren)
        index = 0
        avg = self.getAvg(node, scalar)
        best = avg[index]['child']
        while len(avg[index]['child']) <= 0:
            index+=1
            best = avg[index]['child'] 
        return random.choice(best)
    def getAvg(self, node, scalar=0):
        avg = {'left': [], 'up': [], 'right': [], 'down': []}
        child = {'left': [], 'up': [], 'right': [], 'down': []}
        def avgValue(value):
            if len(value) == 0: return 0
            return sum(value) / len(value)
  
        for c in node.children:
            score = c.reward/c.visits + scalar* math.sqrt(2.0*math.log(node.visits)/float(c.visits))
            if c.state.action == 0:
                avg['left'].append(score)
                child['left'].append(c)
            if c.state.action == 1:
                avg['up'].append(score)
                child['up'].append(c)
            if c.state.action == 2:
                avg['right'].append(score)
                child['right'].append(c)
            if c.state.action == 3:
                avg['down'].append(score)
                child['down'].append(c)
        
        leftAvg = {'action':0, 'avg': avgValue(avg['left']), 'child':child['left']}
        upAvg = {'action':1, 'avg': avgValue(avg['up']), 'child':child['up']}
        rightAvg = {'action':2, 'avg': avgValue(avg['right']), 'child':child['right']}
        downAvg = {'action':3, 'avg': avgValue(avg['down']), 'child':child['down']}
        return [leftAvg, upAvg, rightAvg, downAvg]

    def RANKCHILD(self, node, scalar):
        # for c in node.children:
        #     exploit=c.reward/c.visits
        #     explore=math.sqrt(2.0*math.log(node.visits)/float(c.visits))
        #     score=exploit+scalar*explore
        #     if score==bestscore:
        #         bestchildren.append(c)
        #     if score>bestscore:
        #         bestchildren=[c]
        #         bestscore=score
        return sorted(self.getAvg(node), key=lambda c : c['avg'], reverse=True)
        # return sorted(node.children, key=lambda c : c.reward/c.visits, reverse=True )
    def EXPAND(self, node):
        #현재 노드의 자식을 구한다.
        #현재 노드의 자식과 중복되지 않은 새로운 상태를 만든다.
        while node.fully_expanded() == False:
            tried_children=[c.state for c in node.children]
            new_state, reward = node.state.createPossibleRandomCaseChildState(tried_children)
            node.add_child(new_state, reward)

        #자식을 반환한다
        return node.children[-1]
        
    def BACKUP(self, node, reward):
        while node!=None:
            node.visits+=1
            node.reward+=reward
            node=node.parent

    def DEFAULTPOLICY(self, state):
        # print(state.isEndGame())
        # print()
        copyState = copy.deepcopy(state)
        reward = 0
        while copyState.isEndGame()==False:
            action = copyState.getPossibleRandomAction()
            reward += copyState.step(action)
            copyState.generateNum()
            
            # state.print_table()
            
            # print(reward)
        return reward
        # action = state.getPossibleRandomAction()
        # if action == -1:
        #     return 0
        # reward = state.step(action)
        # return reward