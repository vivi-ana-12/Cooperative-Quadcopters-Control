from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Utils import GraphVisualizer, ExcelFileManager
from .Swarm import Swarm
import traceback


class Simulation:
    def __init__(self,load,trajectoryType,trajectoryNumber):
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
        self.run_simulation();

    def sysCall_cleanup(self,dataset,quadcopters_number,load,filename,trajectoryType,saveGraph): 
        self.sim.stopSimulation(); # Stop and disconnect communication with CoppeliaSim
        print('Disconnected')
        print('Saving file')
        self.excelFileManager.exportDataset(self.swarm.dataset,self.swarm.load,self.swarm.trajectoryType,"11") # Save simulation data -------> IMPORTANT
        self.graphVisualizer.plotDataset("11",load, saveGraph, trajectoryType)
        print('Program ended') 
        
    def run_simulation(self):
        while (True):
            try:
                if self.swarm.quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
                    self.sysCall_cleanup(self.swarm.dataset,4,self.swarm.load,self.swarm.filename,self.swarm.trajectoryType,True)
                    break
            
                self.swarm.update_simulation()
                self.swarm.updateDataSet(); # Save the step data to the dataset
        
                self.client.step();
        
            except: #If an exception occurs, end the program
                self.sysCall_cleanup(self.swarm.dataset,4,self.swarm.load,self.swarm.filename,self.swarm.trajectoryType,True)
                traceback.print_exc() # Print the exception message
                break