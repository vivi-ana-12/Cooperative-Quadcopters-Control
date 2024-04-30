from Scripts.VisualizeGraphs import readAllSheets
from .Quadcopter import Quadcopter
from .ANN import ANN
from .CooperativeMultiAgentGraph import CooperativeMultiAgentGraph
import numpy as np

SIMPLE_SIMULATION = 1
SIMULATION_WITH_OPTIMIZER = 2
SIMULATION_WITH_COOPERATIVE_TRAJECTORY = 3
COMPLETE_SIMULATION = 4

class Swarm: 
    
    def __init__(self,sim,size,load,trajectoryType,trajectoryNumber,simulationMode):
        self.size = size
        self.sim = sim
        self.load = load
        self.trajectoryType = trajectoryType
        self.trajectoryNumber = trajectoryNumber
        self.unloadedModel = ANN()
        self.delay = 16 # Number of delays of the prediction model
        self.quadcopters = [Quadcopter(self.sim, f"Quadcopter[{drone}]", self.delay, self.unloadedModel) for drone in range(self.size)]
        self.simulationMode = simulationMode
        
        
        if self.trajectoryType:     
            self.filename = ".\\DataBase\\Test trajectories\\TestTrajectories.xlsx"    
        else: 
            self.filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories.xlsx"    

        self.results, self.sheets = readAllSheets(self.filename) 
          
        self.trajectory = self.results[self.trajectoryNumber]

        self.set_all_positions_and_frames()
        self.get_initial_positions_matrix()
        self.get_relative_initial_distances()
        self.cooperativeMultiAgentGraph = CooperativeMultiAgentGraph(self.size, self.initialPosition, self.relativeDistances,3)
        
    def set_all_positions_and_frames(self):
        self.update_simulation()
        for quadcopter in range (self.size):
            self.quadcopters[quadcopter].set_initial_state()

    def update_simulation(self):
        for quadcopter in range(self.size):
            self.quadcopters[quadcopter].get_parameters()
            self.quadcopters[quadcopter].set_controller()
            self.quadcopters[quadcopter].set_velocities()
            
            if self.quadcopters[0].t > 0.1:
                self.update_target_positions(quadcopter)
            
            self.quadcopters[quadcopter].trajectory = self.quadcopters[quadcopter].targetPos

    def update_target_positions(self,quadcopter):
        
        if(self.simulationMode == SIMULATION_WITH_OPTIMIZER or self.simulationMode == SIMPLE_SIMULATION):
            if float(round(self.quadcopters[0].t, 2)).is_integer(): # Every second the target positions of the x, y, and z axes are updated
                self.update_trajectory_from_file(quadcopter)
        elif(self.simulationMode == SIMULATION_WITH_COOPERATIVE_TRAJECTORY):
            self.update_trayectory_from_graph(quadcopter)
        else:
            return
        
        self.sim.setObjectPosition(self.quadcopters[quadcopter].targetObj,self.sim.handle_world,[self.quadcopters[quadcopter].targetPos[0],self.quadcopters[quadcopter].targetPos[1],self.quadcopters[quadcopter].targetPos[2]])

    def update_trayectory_from_graph(self, quadcopter):

        if(quadcopter == 0):
            target = (self.quadcopters[0].initPos + self.trajectory.iloc[int(round(self.quadcopters[0].t-1)), [0, 1, 2]]).tolist()
            self.calculatedPositions = self.cooperativeMultiAgentGraph.calculatePosition(target)
        self.quadcopters[quadcopter].targetPos = self.calculatedPositions[:, quadcopter]

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
                self.quadcopters[quadcopter].calculate_loaded_position_error()
                
    def get_initial_positions_matrix(self, axis = np.array([1,1,1])):
        initialPosition = np.zeros((np.sum(axis), self.size))
        for quadcopter in range (self.size):
            initialPosition[:, quadcopter] = self.quadcopters[quadcopter].initPos[axis.astype(bool)]
        
        self.initialPosition = initialPosition
    
    def get_relative_initial_distances(self):
        
        positions = np.array(self.initialPosition)
        relations = np.array([[1.,1.,1.,1.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
        dot_product = np.dot(positions, relations)
        distances = positions - dot_product
        diagonal = np.array([np.diag(distances[0]),np.diag(distances[1]),np.diag(distances[2])])
        m, n = relations.shape   
        mask = np.zeros((3, m, n), dtype=int)

        for axis in range(3):
            for i in range(n):
                mask[axis, :, i] = relations[i]
    
        self.relativeDistances = np.array([np.dot(diagonal[0],mask[0]), np.dot(diagonal[1],mask[1]), np.dot(diagonal[2],mask[2])])

            

        