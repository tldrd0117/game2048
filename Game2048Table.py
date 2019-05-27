import random
import copy
import numpy as np
class TableState:
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    action = -1
    MAX_SCORE = 131070
    def __init__(self):
        self.width = 4
        self.height = 4
        self.table = [[0] * self.width for i in range(self.height)]
        self.done = False

    def reset(self):
        self.table = [[0] * self.width for i in range(self.height)]
        self.done = False


    def rule(self):
        leftMoving = False
        rightMoving = False
        upMoving = False
        downMoving = False

        for i in range(self.height):
            for j in range(self.width):
                if j+1 < self.width and self.table[i][j] == self.table[i][j+1] and self.table[i][j] != 0 and self.table[i][j+1] != 0:
                    leftMoving = True
                    rightMoving = True
                if self.table[i][j] != 0 :
                    if j+1 < self.width and self.table[i][j+1] == 0 :
                        rightMoving = True
                    if j+2 < self.width and self.table[i][j+2] == 0 :
                        rightMoving = True
                    if j+3 < self.width and self.table[i][j+3] == 0 :
                        rightMoving = True
                if self.table[i][j] == 0:
                    if j+1 < self.width and self.table[i][j+1] != 0 :
                        leftMoving = True
                    if j+2 < self.width and self.table[i][j+2] != 0 :
                        leftMoving = True
                    if j+3 < self.width and self.table[i][j+3] != 0 :
                        leftMoving = True


        for j in range(self.width):
            for i in range(self.height):
                if i+1 < self.height and self.table[i][j] == self.table[i+1][j] and self.table[i][j] != 0 and self.table[i+1][j] != 0:
                    upMoving = True
                    downMoving = True
                if self.table[i][j] != 0:
                    if i+1 < self.height and self.table[i+1][j] == 0 :
                        downMoving = True
                    if i+2 < self.height and self.table[i+2][j] == 0 :
                        downMoving = True
                    if i+3 < self.height and self.table[i+3][j] == 0 :
                        downMoving = True
                if self.table[i][j] == 0 :
                    if i+1 < self.height and self.table[i+1][j] != 0 :
                        upMoving = True
                    if i+2 < self.height and self.table[i+2][j] != 0 :
                        upMoving = True
                    if i+3 < self.height and self.table[i+3][j] != 0 :
                        upMoving = True
        return leftMoving, upMoving, rightMoving, downMoving

    def filterImpossibleAction(self, action):
        left, up, right, down = self.rule()
        print(left, up, right, down)

        if left:
            if action == 0:
                return 0
        if up:
            if action == 1:
                return 1
        if right:
            if action == 2:
                return 2
        if down:
            if action == 3:
                return 3
        return -1
    
    def isPossibleAction(self, action):
        left, up, right, down = self.rule()
        if left and action == 0:
            return True
        if up and action == 1:
            return True
        if right and action == 2:
            return True
        if down and action == 3:
            return True
        return False

    def isEndGame(self):
        left, up, right, down = self.rule()
        if not left and not right and not down and not up:
            self.done = True
            return True
        return False


    def generateNum(self):
        emptyList = self.getEmptySpaceNum()
        if len(emptyList) != 0:
            randomNum = list(random.sample(emptyList, 1)[0])
            row = int(randomNum[0])
            col = int(randomNum[1])

            if random.random() < 0.9 : num = 2
            else: num = 4
            self.table[row][col] = num

    def getEmptySpaceNum(self):
        emptyList = list()
        for i in range(self.width):
            for j in range(self.height):
                if self.table[i][j] == 0:
                    emptyList.append([i, j])
        return emptyList

    def upTableDeleteEmptySpace(self, col):
        numberList = list()
        for j in range(self.height):
            if self.table[j][col] != 0:
                numberList.append(self.table[j][col])
                self.table[j][col] = 0

        for k in range(len(numberList)):
            self.table[k][col] = numberList[k]

    def leftTableDeleteEmptySpace(self, row):
        numberList = list()
        for j in range(self.width):
            if self.table[row][j] != 0:
                numberList.append(self.table[row][j])
                self.table[row][j] = 0

        for k in range(len(numberList)):
            self.table[row][k] = numberList[k]

    def downTableDeleteEmptySpace(self, col):
        numberList = list()
        for j in range(self.height):
            if self.table[j][col] != 0:
                numberList.append(self.table[j][col])
                self.table[j][col] = 0

        for k in range(len(numberList)):
            self.table[self.height - len(numberList) + k][col] = numberList[k]

    def rightTableDeleteEmptySpace(self, row):
        numberList = list()
        for j in range(self.height):
            if self.table[row][j] != 0:
                numberList.append(self.table[row][j])
                self.table[row][j] = 0

        for k in range(len(numberList)):
            self.table[row][self.height - len(numberList) + k] = numberList[k]


    def upTableMergeSpace(self, col):
        mergeCount = 0
        mergeScore = 0
        for j in range(self.height-1):
            if self.table[j][col] == 0:
                continue
            if self.table[j][col] == self.table[j+1][col]:
                self.table[j][col] = self.table[j][col]*2
                self.table[j+1][col] = 0
                self.upTableDeleteEmptySpace(col)
                mergeCount+=1
                mergeScore = self.table[j][col]

        if mergeCount > 0:
            return mergeCount, mergeScore
        return 0, 0
            # self.upTableMergeSpace(col)

    def leftTableMergeSpace(self, row):
        mergeCount = 0
        mergeScore = 0
        for j in range(self.width-1):
            if self.table[row][j] == 0:
                continue
            if self.table[row][j] == self.table[row][j+1]:
                self.table[row][j] = self.table[row][j]*2
                self.table[row][j+1] = 0
                self.leftTableDeleteEmptySpace(row)
                mergeCount+=1
                mergeScore = self.table[row][j]

        if mergeCount > 0:
            return mergeCount, mergeScore
        return 0, 0
            # self.leftTableMergeSpace(row)

    def downTableMergeSpace(self, col):
        mergeCount = 0
        mergeScore = 0
        for j in reversed(range(1, self.height)):
            if self.table[j][col] == 0:
                continue
            if self.table[j][col] == self.table[j-1][col]:
                self.table[j][col] = self.table[j][col]*2
                self.table[j-1][col] = 0
                self.downTableDeleteEmptySpace(col)
                mergeCount+=1
                mergeScore+=self.table[j][col]

        if mergeCount > 0:
            return mergeCount, mergeScore
        return 0, 0
            # self.downTableMergeSpace(col)

    def rightTableMergeSpace(self, row):
        mergeCount = 0
        mergeScore = 0
        for j in reversed(range(1, self.height)):
            if self.table[row][j] == 0:
                continue
            if self.table[row][j] == self.table[row][j-1]:
                self.table[row][j] = self.table[row][j]*2
                self.table[row][j-1] = 0
                self.rightTableDeleteEmptySpace(row)
                mergeCount+=1
                mergeScore = self.table[row][j]

        if mergeCount > 0:
            return mergeCount, mergeScore
        return 0, 0
        #     self.rightTableMergeSpace(row)

    def up(self):
        rewardCountSum = 0
        rewardScoreSum = 0
        for i in range(self.width):
            self.upTableDeleteEmptySpace(i)
            rewardCount, rewardScore = self.upTableMergeSpace(i)
            rewardCountSum += rewardCount
            rewardScoreSum += rewardScore
        if rewardCount > 0:
            return rewardCountSum, rewardScoreSum
        return 0, 0

    def left(self):
        rewardCountSum = 0
        rewardScoreSum = 0
        for i in range(self.height):
            self.leftTableDeleteEmptySpace(i)
            rewardCount, rewardScore = self.leftTableMergeSpace(i)
            rewardCountSum += rewardCount
            rewardScoreSum += rewardScore
        if rewardCount > 0:
            return rewardCountSum, rewardScoreSum
        return 0, 0

    def down(self):
        rewardCountSum = 0
        rewardScoreSum = 0
        for i in range(self.width):
            self.downTableDeleteEmptySpace(i)
            rewardCount, rewardScore = self.downTableMergeSpace(i)
            rewardCountSum += rewardCount
            rewardScoreSum += rewardScore

        if rewardCount > 0:
            return rewardCountSum, rewardScoreSum
        return 0, 0

    def right(self):
        rewardCountSum = 0
        rewardScoreSum = 0
        for i in range(self.height):
            self.rightTableDeleteEmptySpace(i)
            rewardCount, rewardScore = self.rightTableMergeSpace(i)
            rewardCountSum += rewardCount
            rewardScoreSum += rewardScore

        if rewardCount > 0:
            return rewardCountSum, rewardScoreSum
        return 0, 0

    def step(self, action):
        rewardCount = 0
        rewardScore = 0
        if action == 0:
            rewardCount, rewardScore = self.left()
        elif action == 1:
            rewardCount, rewardScore = self.up()
        elif action == 2:
            rewardCount, rewardScore = self.right()
        elif action == 3:
            rewardCount, rewardScore = self.down()
        else:
            print("no step")
        tableSum = np.sum(self.table)
        return 1.0 - (self.MAX_SCORE-tableSum)/self.MAX_SCORE
        # if reward > 0:
            # return 1
        # else:
            # return -1
    
    def move_case(self):
        left, up, right, down = self.rule()
        moveCases = [0,0,0,0]
        if left:
            moveCases[0]+=self.get_action_case_num(0)
        if up:
            moveCases[1]+=self.get_action_case_num(1)
        if right:
            moveCases[2]+=self.get_action_case_num(2)
        if down:
            moveCases[3]+=self.get_action_case_num(3)
        return sum(moveCases)   
    
    def get_action_case_num(self, action):
        if not self.isPossibleAction(action):
            return 0
        before_table = copy.deepcopy(self.table)
        self.step(action)
        empty_num = len(self.getEmptySpaceNum())
        self.table = before_table
        return empty_num * 2 #2 와 4 가능 
    
    def createPossibleRandomCaseChildState(self, tried_child):
        leftAction = []
        upAction = []
        rightAction = []
        downAction = []
        for childNode in tried_child:
            if childNode.action == self.LEFT:
                leftAction.append(childNode.table)
            if childNode.action == self.UP:
                upAction.append(childNode.table)
            if childNode.action == self.RIGHT:
                rightAction.append(childNode.table)
            if childNode.action == self.DOWN:
                downAction.append(childNode.table)

        newState = TableState()
        fully_left_case_num = self.get_action_case_num(self.LEFT)
        fully_up_case_num = self.get_action_case_num(self.UP)
        fully_right_case_num = self.get_action_case_num(self.RIGHT)
        fully_down_case_num = self.get_action_case_num(self.DOWN)
        
        newState.table = copy.deepcopy(self.table)

        reward = 0
        #무언가 잘못된 것이 있음
        if fully_left_case_num !=0 and fully_left_case_num > len(leftAction):
            newState, reward = self.tryAction(self.LEFT, leftAction, newState)
        elif fully_up_case_num !=0 and fully_up_case_num > len(upAction):
            newState, reward = self.tryAction(self.UP, upAction, newState)
        elif fully_right_case_num !=0 and fully_right_case_num > len(rightAction):
            newState, reward = self.tryAction(self.RIGHT, rightAction, newState)
        elif fully_down_case_num !=0 and fully_down_case_num > len(downAction):
            newState, reward = self.tryAction(self.DOWN, downAction, newState)
        # newState.action = -1
        return newState, reward

    def tryAction(self, action, actionList, newState):
        empty_nums_two, empty_nums_four = self.get_possible_empty_num(action, actionList)
        if len(empty_nums_two) > 0:
            reward = newState.setStateWithActionAndEmptyNum(action, empty_nums_two[0], 2)
        else:
            reward = newState.setStateWithActionAndEmptyNum(action, empty_nums_four[0], 4)
        return newState, reward
    
    def setStateWithActionAndEmptyNum(self, action, empty_num, num):
        reward = self.step(action)
        self.action = action
        self.generateNumFromIndex(empty_num, num)
        return reward
    
    def generateNumFromIndex(self, empty_num, num):
        self.table[empty_num[0]][empty_num[1]] = num

    #이미 추가한 빈값이 같으면 테이블 삭제
    def get_possible_empty_num(self, action, tables):
        before_table = copy.deepcopy(self.table)
        self.step(action)
        empty_nums_two = self.getEmptySpaceNum()
        empty_nums_four = self.getEmptySpaceNum()

        if len(tables) == 0:
            self.table = before_table
            return empty_nums_two, empty_nums_four

        results_two = []
        results_four = []
        for empty_num in empty_nums_two:
            for table in tables:
                if table[empty_num[0]][empty_num[1]] == 2 and self.table[empty_num[0]][empty_num[1]] != 2:
                    isSame = True
                    for i in range(0,4):
                        for j in range(0,4):
                            if(empty_num[0] == i and empty_num[1] == j):
                                continue
                            isSame = isSame and self.table[i][j] == table[i][j]
                    if isSame:
                        results_two.append([empty_num[0], empty_num[1]])

        for empty_num in empty_nums_four:
            for table in tables:
                if table[empty_num[0]][empty_num[1]] == 4 and self.table[empty_num[0]][empty_num[1]] != 4:
                    isSame = True
                    for i in range(0,4):
                        for j in range(0,4):
                            if(empty_num[0] == i and empty_num[1] == j):
                                continue
                            isSame = isSame and self.table[i][j] == table[i][j]
                    if isSame:
                        results_four.append([empty_num[0], empty_num[1]])
        for result in results_two:
            empty_nums_two.remove(result)
        for result in results_four:
            empty_nums_four.remove(result)

        self.table = before_table
        return empty_nums_two, empty_nums_four
    
    def getPossibleRandomAction(self):
        left, up, right, down = self.rule()
        actions = []
        if left:
            actions.append(0)
        if up:
            actions.append(1)
        if right:
            actions.append(2)
        if down:
            actions.append(3)
        if self.isEndGame():
            return -1

        return random.choice(actions)

    def print_table(self):
        print("------------------------")
        for i in range(4):
            print("%5d %5d %5d %5d"%(self.table[i][0], self.table[i][1], self.table[i][2], self.table[i][3]))
        print("------------------------")

