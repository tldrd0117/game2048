class MCTSNode:
    def __init__(self, state, parent=None):
        self.visits=1
        self.reward=0.0
        self.state=state
        self.children=[]
        self.simulation=[]
        self.parent=parent
    def add_child(self,child_state, reward):
        child = MCTSNode(child_state, self)
        child.reward = reward
        self.children.append(child)
    def update(self,reward):
        self.reward+=reward
        self.visits+=1
    def fully_expanded(self):
        if len(self.children)==self.state.move_case():
            return True
        return False
    def __repr__(self):
        s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
        return s
