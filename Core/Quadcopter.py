import numpy as np

class Quadcopter:
        
    propellerHandles = []
    jointHandles = []
    particleObjects = [-1,-1,-1,-1]
    
    pParam=0.5
    iParam=0
    dParam=0
    vParam=-2

    cumul=0
    lastE=0
    pAlphaE=0
    pBetaE=0
    psp2=0
    psp1=0
    
    t = 0
    e = 0

    prevEuler=0

    Kpalpha = 0.1
    Kialpha = 0.002
    Kdalpha = 0.002
    
    epoch = 0 

    def __init__(self,sim,name):
        self.name = name
        self.targetObj = sim.getObject("/"+str(name)+'/target')
        self.d = sim.getObject("/"+str(name)+'/base')

        self.heli = sim.getObject("/"+str(name))
        
        for i in range(4):
            self.propellerHandles.append(sim.getObject('./propeller['+str(i)+']/respondable'))
            self.jointHandles.append(sim.getObject('./propeller['+str(i)+']/joint'))

    
        self.objectHandle = sim.getObject("./"+str(name));
        self.scriptHandle = sim.getScript(1,self.objectHandle);
        self.sim = sim 

        self.sim.setObjectParent(self.targetObj,-1,True)
        
        self.kP = []
        self.kI = []
        self.kD = []

    def get_parameters(self):
        self.pos,self.targetPos,self.l,self.sp,self.m,self.euler,self.vx,self.vy,self.t = self.sim.callScriptFunction('getParameters',self.scriptHandle,self.d,self.targetObj,self.heli);
        self.targetPos = np.array(self.targetPos)

    def set_controller(self):
        # Vertical control:
        self.e = (self.targetPos[2]-self.pos[2]) #Compare the current position of the quadcopter in the z axis with the desired position
        self.cumul = self.cumul + self.e
        self.pv = self.pParam*self.e
        self.thrust = 5.45+self.pv+self.iParam*self.cumul+self.dParam*(self.e-self.lastE)+self.l[2]*self.vParam
        self.lastE = self.e
        
        # Horizontal control: 
        self.alphaE = (self.vy[2]-self.m[11])
        self.alphaCorr = 1*self.alphaE+2.5*(self.alphaE-self.pAlphaE)
        self.betaE = (self.vx[2]-self.m[11])
        self.betaCorr = -1*self.betaE-2.5*(self.betaE-self.pBetaE)
        self.pAlphaE = self.alphaE
        self.pBetaE = self.betaE
        self.alphaCorr = self.alphaCorr+self.sp[1]*0.005+1*(self.sp[1]-self.psp2)
        self.betaCorr = self.betaCorr-self.sp[0]*0.005-1*(self.sp[0]-self.psp1)
        self.psp2 = self.sp[1]
        self.psp1 = self.sp[0]
                
        # Rotational control:
        self.rotCorr = self.euler[2]*0.1+2*(self.euler[2]-self.prevEuler)
        self.prevEuler = self.euler[2]
        
    # def set_ANN_controller(self, positionError, dataset, du):

    #     dy = np.array(dataset[:3])[:, -1]-np.array(dataset[:3])[:, -2]

    #     for axis in range(2, 3):  # range(1):  # 2,3
    #         self.kP[axis] = self.kP[axis] + self.Kpalpha * (positionError[axis][-1]**2) * (dy[axis]/du[axis] if du[axis] != 0 else 1)
    #         if self.epoch >= 2:
    #             self.kD[axis] = self.kD[axis] + self.Kdalpha * positionError[axis][-1] * (positionError[axis][-1] - positionError[axis][-2]) * (dy[axis]/du[axis] if du[axis] != 0 else 1)
    #         if self.epoch >= 4:
    #             self.kI[axis] = self.kI[axis] + positionError[axis][-1] * (self.Kialpha * np.sum(np.array(positionError[axis][-4:]))) * positionError[axis][-1] * (dy[axis]/du[axis] if du[axis] != 0 else 1)

    #     self.epoch = self.epoch + 1
    
    def set_ANN_controller(self,positionError,dataset):
        
        # dy = np.array(dataset[:3])[:, -1]-np.array(dataset[:3])[:, -2]
        # du = np.array(dataset[4:7])[:,-1]-np.array(dataset[4:7])[:, -2]
        
        for axis in range (2,3): 
           # self.kP[axis] = self.kP[axis] + self.Kpalpha * (positionError[axis][-1]**2) * (dy[axis]/du[axis] if du[axis] != 0 else 1)

            self.kP[axis] = self.kP[axis] + self.Kpalpha * (positionError[axis][-1])
            if self.epoch >= 2:
                self.kD[axis] = self.kD[axis] + self.Kdalpha*(positionError[axis][-1]-positionError[axis][-2])
            if self.epoch >= 4: 
                self.kI[axis] = self.kI[axis] + (self.Kialpha*(positionError[axis][-1]+positionError[axis][-2]+positionError[axis][-3]+positionError[axis][-4]))
        
        self.epoch = self.epoch + 1

    def set_velocities(self):
        self.sim.callScriptFunction('setVelocities',self.scriptHandle,self.thrust,self.alphaCorr,self.betaCorr,self.rotCorr)