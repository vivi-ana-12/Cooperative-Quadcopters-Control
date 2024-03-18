from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from Classes import Swarm
import traceback
from Utils import GraphVisualizer, ExcelFileManager

def sysCall_cleanup(sim,dataset,quadcopters_number,load,filename,trajectoryType,saveGraph): 
    sim.stopSimulation(); # Stop and disconnect communication with CoppeliaSim
    print('Disconnected')
    print('Saving file')
    excelFileManager.exportDataset(dataset,load,trajectoryType,"11") # Save simulation data -------> IMPORTANT
    graphVisualizer.plotDataset("11",load, saveGraph, trajectoryType)
    print('Program ended') 

def updateDataSet(quadcopters, dataset):
    # Obtener los datos de posición, posición objetivo, errores, tiempo, empuje y correcciones de todos los drones
    data = zip(
        [quadcopter.pos[0] for quadcopter in quadcopters],  # Posición actual 
        [quadcopter.pos[1] for quadcopter in quadcopters],  # Posición actual
        [quadcopter.pos[2] for quadcopter in quadcopters],  # Posición actual
        [quadcopter.targetPos[0] for quadcopter in quadcopters],  # Posición objetivo
        [quadcopter.targetPos[1] for quadcopter in quadcopters],  # Posición objetivo
        [quadcopter.targetPos[2] for quadcopter in quadcopters],  # Posición objetivo
        [quadcopter.betaE for quadcopter in quadcopters],  # Error en x
        [quadcopter.alphaE for quadcopter in quadcopters],  # Error en y
        [quadcopter.e for quadcopter in quadcopters],  # Error en z
        [quadcopter.t for quadcopter in quadcopters],  # Tiempo
        [quadcopter.thrust for quadcopter in quadcopters],  # Empuje
        [quadcopter.betaCorr for quadcopter in quadcopters],  # Corrección beta
        [quadcopter.rotCorr for quadcopter in quadcopters]  # Corrección rot
    )

    # Guardar los datos en el dataset
    for drone_data, dataset_entry in zip(data, dataset):
        for value, entry in zip(drone_data, dataset_entry):
            entry.append(value)
            
# ------------------------------------------------------------------------------------------------------------------
print('Program started\n')
client = RemoteAPIClient(); # Start RemoteApp connection client
sim = client.getObject('sim');  # Retrieve the object handle
client.setStepping(True); # Activate staggered mode
print('Connected\n')

sim.startSimulation(); 
client.step(); # One step in the simulation
load = True # Indicates if the Coppelia simulation is with Load or without Load 
trajectoryType = True # Training False, Test True 
trajectoryNumber = "11"

swarm = Swarm(sim,4,load,trajectoryType,trajectoryNumber)
dataset = [[[] for x in range(13)] for y in range(swarm.size)] #Create the dataset with 13 positions for each drone

graphVisualizer = GraphVisualizer(4)
excelFileManager = ExcelFileManager()
while (True):
    try:
        if swarm.quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
            sysCall_cleanup(sim,dataset,4,swarm.load,swarm.filename,swarm.trajectoryType,True)
            break
    
        swarm.update_simulation()
        updateDataSet(swarm.quadcopters,dataset); # Save the step data to the dataset

        client.step();

    except: #If an exception occurs, end the program
        sysCall_cleanup(sim,dataset,4,load,swarm.filename,trajectoryType,True)
        traceback.print_exc() # Print the exception message
        break

