from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Utils import GraphVisualizer, ExcelFileManager
from .Swarm import Swarm
import traceback

SIMPLE_SIMULATION = 1
SIMULATION_WITH_OPTIMIZER = 2
SIMULATION_WITH_COOPERATIVE_TRAJECTORY = 3
COMPLETE_SIMULATION = 4

SIMULATION_MODES = {
    SIMPLE_SIMULATION: "simple simulation: no optimizer controller, trajectory from input file (no consensus).",
    SIMULATION_WITH_OPTIMIZER: "simulation with optimizer: optimizer controller for each drone, trajectory from input file (no consensus).",
    SIMULATION_WITH_COOPERATIVE_TRAJECTORY: "simulation with cooperative trajectory: no optimizer controller but with consensus trajectory.",
    COMPLETE_SIMULATION: "complete simulation: optimizer controller for each drone, trajectory from swarm consensus (graphs)."
}


class Simulation:
    def __init__(self, simulationMode,trajectoryType = True, trajectoryNumber = '1', load = True):
        
        self.simulationMode = simulationMode
        message = SIMULATION_MODES.get(self.simulationMode)
        if message is None:
            print("Invalid simulation mode.")
            exit()
        print('Running',message)
        
        self.client = RemoteAPIClient(); # Start RemoteApp connection client
        self.sim = self.client.getObject('sim');
        self.client.setStepping(True); # Activate staggered mode
        print('Connected\n')
        
        self.graphVisualizer = GraphVisualizer(4)
        self.excelFileManager = ExcelFileManager()
        self.start_simulation(load,trajectoryType,trajectoryNumber)


    def start_simulation(self,load,trajectoryType,trajectoryNumber):
        print('Simulation started\n')
        self.sim.startSimulation();
        self.client.step();
        self.swarm = Swarm(self.sim,4,load,trajectoryType,trajectoryNumber)
        self.dataset = [[[] for x in range(13)] for y in range(self.swarm.size)] #Create the dataset with 13 positions for each drone

        self.iteration = 0
        self.run_simulation();

    def sysCall_cleanup(self): 
        self.sim.stopSimulation(); # Stop and disconnect communication with CoppeliaSim
        print('Disconnected')
        print('Saving file')
        if self.simulationMode == SIMULATION_WITH_OPTIMIZER or self.simulationMode == COMPLETE_SIMULATION: 
            self.excelFileManager.exportCompleteDataset(self.dataset,self.swarm.load,self.swarm.trajectoryType,self.swarm.trajectoryNumber,self.allPredictions)
            self.graphVisualizer.plotCompleteDataset(self.swarm.trajectoryNumber,self.swarm.load, True, self.swarm.trajectoryType)

        elif self.simulationMode == SIMPLE_SIMULATION:
            self.excelFileManager.exportSimpleDataset(self.dataset,self.swarm.load,self.swarm.trajectoryType,self.swarm.trajectoryNumber)
            self.graphVisualizer.plotSimpleDataset(self.swarm.trajectoryNumber,self.swarm.load, True, self.swarm.trajectoryType)

        print('Program ended') 
    
    
    def update_dataSet(self):
        data = zip(
            [quadcopter.pos[0] for quadcopter in self.swarm.quadcopters],         # Posición actual x
            [quadcopter.pos[1] for quadcopter in self.swarm.quadcopters],         # Posición actual y
            [quadcopter.pos[2] for quadcopter in self.swarm.quadcopters],         # Posición actual z
            [quadcopter.trajectory[0] for quadcopter in self.swarm.quadcopters],  # Trayectoria x
            [quadcopter.trajectory[1] for quadcopter in self.swarm.quadcopters],  # Trayectoria y
            [quadcopter.trajectory[2] for quadcopter in self.swarm.quadcopters],  # Trayectoria z
            [quadcopter.betaE for quadcopter in self.swarm.quadcopters],          # Error en x
            [quadcopter.alphaE for quadcopter in self.swarm.quadcopters],         # Error en y
            [quadcopter.e for quadcopter in self.swarm.quadcopters],              # Error en z
            [quadcopter.t for quadcopter in self.swarm.quadcopters],              # Tiempo
            [quadcopter.thrust for quadcopter in self.swarm.quadcopters],         # Empuje
            [quadcopter.betaCorr for quadcopter in self.swarm.quadcopters],       # Corrección beta
            [quadcopter.rotCorr for quadcopter in self.swarm.quadcopters]         # Corrección rot
        )
    
        for drone_data, dataset_entry in zip(data, self.dataset):
            for value, entry in zip(drone_data, dataset_entry):
                entry.append(value)
                
    def run_simulation(self):
        self.allPredictions = []

        while (True):
            try:
                if self.swarm.quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
                    if self.simulationMode == SIMULATION_WITH_OPTIMIZER or self.simulationMode == COMPLETE_SIMULATION:
                        for quadcopter in range (4):
                            quad_data = [[0]*17 + self.swarm.quadcopters[quadcopter].unloaded_behavior_predictions[col].tolist() for col in self.swarm.quadcopters[quadcopter].unloaded_behavior_predictions.columns]
                            self.allPredictions.append(quad_data)
                            
                    self.sysCall_cleanup()
                    break
            
                self.swarm.update_simulation()

                if self.iteration%2 == 0 and (self.simulationMode == SIMULATION_WITH_OPTIMIZER or self.simulationMode == COMPLETE_SIMULATION) :
                    self.update_dataSet(); # Save the step data to the dataset
                    self.swarm.predict_quadcopters_behavior(self.iteration)
                
                if self.simulationMode == SIMPLE_SIMULATION:
                    self.update_dataSet(); # Save the step data to the dataset

                self.iteration = self.iteration + 1
                self.client.step();
        
            except: #If an exception occurs, end the program
                self.sim.stopSimulation(); # Stop and disconnect communication with CoppeliaSim
                traceback.print_exc() # Print the exception message
                break