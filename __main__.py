from Core import Simulation

            
load = True # Indicates if the Coppelia simulation is with Load or without Load 
trajectoryType = True # Training False, Test True 
trajectoryNumber = "11"

simulation = Simulation(load,trajectoryType,trajectoryNumber)
