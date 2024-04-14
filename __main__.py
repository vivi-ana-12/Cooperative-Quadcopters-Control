from Core import Simulation

SIMPLE_SIMULATION = 1
SIMULATION_WITH_OPTIMIZER = 2
SIMULATION_WITH_COOPERATIVE_TRAJECTORY = 3
COMPLETE_SIMULATION = 4

load = True # Indicates if the Coppelia simulation is with Load or without Load 
trajectoryType = True # Training False, Test True 
trajectoryNumber = "1"

simulationMode = SIMPLE_SIMULATION
simulation = Simulation(simulationMode,trajectoryType,trajectoryNumber,load)