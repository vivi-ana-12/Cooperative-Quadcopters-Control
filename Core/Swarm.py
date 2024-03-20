from .Quadcopter import Quadcopter
from Scripts.VisualizeGraphs import readAllSheets

class Swarm: 
    
    def __init__(self,sim,size,load,trajectoryType,trajectoryNumber):
        self.size = size
        self.sim = sim
        self.load = load
        self.trajectoryType = trajectoryType
        self.trajectoryNumber = trajectoryNumber
        self.quadcopters = []
        

        self.quadcopters = [Quadcopter(self.sim, f"Quadcopter[{drone}]") for drone in range(self.size)]
        self.delay = 16 # Number of delays of the prediction model

        if self.trajectoryType:     
            self.filename = ".\\DataBase\\Test trajectories\\TestTrajectories.xlsx"    
        else: 
            self.filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories.xlsx"    

        self.results, self.sheets = readAllSheets(self.filename) 
          
        self.trajectory = self.results[self.trajectoryNumber]

        self.update_simulation()
        self.save_initial_positions()


    def save_initial_positions(self):
        self.initPos=[]
        for quadcopter in range (4):
            self.initPos.append(self.quadcopters[quadcopter].targetPos[:])
        
    def update_simulation(self):
        for drone in range(self.size):
            self.quadcopters[drone].get_parameters()
            self.quadcopters[drone].set_controller()
            self.quadcopters[drone].set_velocities()
            
        if float(round(self.quadcopters[0].t, 2)).is_integer(): # Every second the target positions of the x, y, and z axes are updated
              self.update_target_positions()
              
    def update_target_positions(self):
        for quadcopter in range (self.size):
            self.quadcopters[quadcopter].targetPos[0] = self.initPos[quadcopter][0] + self.trajectory["x"][round(self.quadcopters[0].t-1, 2)] 
            self.quadcopters[quadcopter].targetPos[1] = self.initPos[quadcopter][1] + self.trajectory["y"][round(self.quadcopters[0].t-1, 2)] 
            self.quadcopters[quadcopter].targetPos[2] = self.initPos[quadcopter][2] + self.trajectory["z"][round(self.quadcopters[0].t-1, 2)] 
            
            self.sim.setObjectPosition(self.quadcopters[quadcopter].targetObj,self.sim.handle_world,[self.quadcopters[quadcopter].targetPos[0],self.quadcopters[quadcopter].targetPos[1],self.quadcopters[quadcopter].targetPos[2]])

