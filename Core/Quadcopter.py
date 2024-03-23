import tensorflow as tf
import numpy as np

class Quadcopter:
        
    pParam=0.5
    iParam=0
    dParam=0
    vParam=-2

    Kpalpha = 0.1
    Kialpha = 0.002
    Kdalpha = 0.002
    
    epoch = 0 

    def __init__(self,sim,name,delay,model):
        self.name = name
        self.targetObj = sim.getObject("/"+str(name)+'/target')
        self.d = sim.getObject("/"+str(name)+'/base')

        self.heli = sim.getObject("/"+str(name))
        
        self.propellerHandles = []
        self.jointHandles = []
        self.cumul=0
        self.lastE=0
        
        self.pAlphaE=0
        self.pBetaE=0
        self.psp2=0
        self.psp1=0

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
        
  
        self.e = 0
        self.prevEuler = 0
        
        self.delayedTrajectory = np.zeros((3, delay))
        self.delayedPosition = np.zeros((3, delay))
        
        self.ANN_model = model


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

    def set_velocities(self):
        self.sim.callScriptFunction('setVelocities',self.scriptHandle,self.thrust,self.alphaCorr,self.betaCorr,self.rotCorr)
        
    # def predict_unloaded_behavior(self):
    #     self.update_input_features(actualInputTrajectory, lastPrediction)
        
    def create_input_features(self, actualInputTrajectory, lastPrediction):
        self.update_delay_array(self.delayedTrajectory, actualInputTrajectory)
        self.update_delay_array(self.delayedPosition, lastPrediction)
        
        self.input_features = np.concatenate((self.delayedTrajectory, self.delayedPosition), axis=1)
        
    def update_delay_array(self, array, new_value):
        self.array[:, :-1] = self.array[:, 1:]
        self.array[:, -1] = new_value
        return array
        