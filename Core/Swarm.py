from Scripts.VisualizeGraphs import readAllSheets
from .Quadcopter import Quadcopter
from .ANN import ANN

class Swarm: 
    
    def __init__(self,sim,size,load,trajectoryType,trajectoryNumber):
        self.size = size
        self.sim = sim
        self.load = load
        self.trajectoryType = trajectoryType
        self.trajectoryNumber = trajectoryNumber
        self.unloadedModel = ANN()
        self.delay = 16 # Number of delays of the prediction model
        self.quadcopters = [Quadcopter(self.sim, f"Quadcopter[{drone}]", self.delay, self.unloadedModel) for drone in range(self.size)]
        
        if self.trajectoryType:     
            self.filename = ".\\DataBase\\Test trajectories\\TestTrajectories.xlsx"    
        else: 
            self.filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories.xlsx"    

        self.results, self.sheets = readAllSheets(self.filename) 
          
        self.trajectory = self.results[self.trajectoryNumber]

        self.set_all_positions_and_frames()
        
        
    def set_all_positions_and_frames(self):
        self.update_simulation()
        for quadcopter in range (self.size):
            self.quadcopters[quadcopter].set_initial_state()

            
    def update_simulation(self):
        for quadcopter in range(self.size):
            self.quadcopters[quadcopter].get_parameters()
            self.quadcopters[quadcopter].set_controller()
            self.quadcopters[quadcopter].set_velocities()
            
            if float(round(self.quadcopters[0].t, 2)).is_integer(): # Every second the target positions of the x, y, and z axes are updated
                self.update_target_positions(quadcopter)
            
            self.quadcopters[quadcopter].trajectory = self.quadcopters[quadcopter].targetPos

    def update_target_positions(self,quadcopter):
        self.update_trajectory_from_file(quadcopter)
        self.sim.setObjectPosition(self.quadcopters[quadcopter].targetObj,self.sim.handle_world,[self.quadcopters[quadcopter].targetPos[0],self.quadcopters[quadcopter].targetPos[1],self.quadcopters[quadcopter].targetPos[2]])
            
    def update_trajectory_from_file(self,quadcopter):
        self.quadcopters[quadcopter].targetPos = (self.quadcopters[quadcopter].initPos + self.trajectory.iloc[int(round(self.quadcopters[0].t-1)), [0, 1, 2]]).tolist()
        
    def update_predictions_inputs(self,iteration,quadcopter):
        if iteration <= (16+1)*2:
            self.quadcopters[quadcopter].initialize_delayed_arrays(iteration)
        else: 
            self.quadcopters[quadcopter].update_delayed_arrays()
            
    def predict_quadcopters_behavior(self, iteration):
        for quadcopter in range (self.size):
            self.update_predictions_inputs(iteration,quadcopter)
            if iteration >= (16+1)*2:
                self.quadcopters[quadcopter].predict_unloaded_behavior()


        