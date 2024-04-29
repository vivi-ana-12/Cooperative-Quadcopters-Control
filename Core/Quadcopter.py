import pandas as pd
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
        
        
        self.ANN_model = model
        
        self.unloaded_behavior_predictions = pd.DataFrame()
        self.loaded_position_errors = pd.DataFrame()

        self.delayedTrajectory = pd.DataFrame(np.zeros((17, 3)), columns=['x', 'y', 'z'])
        self.actualInputTrajectory = pd.DataFrame(np.zeros((17, 3)), columns=['x', 'y', 'z'])
        self.trajectory = []

        
    def set_initial_state(self):
        self.initPos = self.targetPos.copy()
        self.ownFramePos = [ 0.738 - self.pos[0],
                            -0.755 - self.pos[1],
                            1.6505 - self.pos[2]]


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
        
    def predict_unloaded_behavior(self):
        input_features = self.create_input_features()
        self.lastPrediction = self.ANN_model.predict(input_features)
        self.unloaded_behavior_predictions = pd.concat([self.unloaded_behavior_predictions,self.lastPrediction - self.ownFramePos], ignore_index=True,sort = False)

    def initialize_delayed_arrays(self,iteration):
        if iteration < 17*2:
            self.update_delay_array(self.delayedTrajectory,np.array(self.pos)+np.array(self.ownFramePos))
        if iteration <= 17*2 and iteration != 0:
            self.update_delay_array(self.actualInputTrajectory,np.array(self.targetPos)+np.array(self.ownFramePos))

    def update_delayed_arrays(self):
        self.update_delay_array(self.delayedTrajectory,self.lastPrediction.values.tolist()[0])
        self.update_delay_array(self.actualInputTrajectory,np.array(self.targetPos)+np.array(self.ownFramePos))
        

    def create_input_features(self):
        concatenated = pd.concat([self.actualInputTrajectory['x'], self.actualInputTrajectory['y'], self.actualInputTrajectory['z'],
                                  self.delayedTrajectory['x'], self.delayedTrajectory['y'], self.delayedTrajectory['z']])
                
        input_features = np.expand_dims(concatenated.values.flatten(), axis=0)
        
        return input_features
    
    def calculate_loaded_position_error(self):
        self.last_loaded_position_error = (self.lastPrediction - self.ownFramePos) - self.pos
        self.loaded_position_errors = pd.concat([self.loaded_position_errors,self.last_loaded_position_error], ignore_index=True,sort = False)

    def update_delay_array(self, array, newData):
        array.iloc[1:, :] = array.iloc[:-1,:].values
        array.loc[0] = newData
                
        return array        