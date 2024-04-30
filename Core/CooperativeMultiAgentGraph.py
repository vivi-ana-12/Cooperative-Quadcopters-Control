import numpy as np

class CooperativeMultiAgentGraph:
    
    K1 = 0.5
    K2 = 0.5
    
    def __init__(self, agentAmout, initialPosition, relativeDistances, axisAmout=2):
        self.agentAmout = agentAmout
        self.X = initialPosition
        self.Px = relativeDistances
        self.axisAmout = axisAmout
        self.setUpMatrix()
        
    def setUpMatrix(self):
        self.X_t = self.X#np.zeros((self.axisAmout, self.agentAmout))
        self.u = np.zeros((self.axisAmout, self.agentAmout))
        self.setLeaderMatrix()      

    def setLeaderMatrix(self):
        #TODO: Fix this
        self.A = np.array([[0,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0]]*self.axisAmout)
        self.A = self.A.reshape((self.axisAmout, self.agentAmout, self.agentAmout))

        self.b = np.array([[1,0,0,0]*self.axisAmout])
        self.b = self.b.reshape((self.axisAmout, self.agentAmout))
    
    def calculatePosition(self, referencePosition):

        for agentId in range(self.agentAmout):
          u1 =  np.sum(self.K1*((self.X_t[:, agentId][:, np.newaxis] - (self.X_t+self.Px[:, agentId,:]))*self.A[:, agentId,:]),axis=1)
          self.u[:,agentId] = u1 + self.K2*self.b[:, agentId]*(self.X_t[:,agentId]-referencePosition)
        self.X_t = self.X_t-self.u
        return self.X_t