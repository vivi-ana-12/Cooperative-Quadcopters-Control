from Classes import Swarm
import traceback
import pandas as pd
from Scripts import VisualizeGraphs
from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def sysCall_cleanup(sim,dataset,quadcopters_number,load,filename,trajectoryType,saveGraph): 
    sim.stopSimulation(); # Stop and disconnect communication with CoppeliaSim
    print('Disconnected')
    print('Saving file')
    saveFile(dataset,load,trajectoryType,"11") # Save simulation data -------> IMPORTANT 
    VisualizeGraphs.plotData("11",load,saveGraph,trajectoryType)
    print('Program ended') 

    
def saveFile(dataset, load, trajectoryType, trajectoryNumber):
    if trajectoryType:
        filename = ".\\DataBase\\Test trajectories\\TestTrajectory_" + str(trajectoryNumber) + "_Results.xlsx"
    else:
        filename = ".\\DataBase\\Training trajectories\\TrainingTrajectory_" + str(trajectoryNumber) + "_Results.xlsx"

    sheets = []

    if not load:
        writer = pd.ExcelWriter(filename, engine='openpyxl')  # Create the document or overwrite it

    for i in range(4):  # Create a sheet for each drone
        sheets.append({'t': dataset[i][9],
                       'x': dataset[i][0], 'y': dataset[i][1], 'z': dataset[i][2],
                       'target x': dataset[i][3], 'target y': dataset[i][4], 'target z': dataset[i][5],
                       'betaE': dataset[i][6], 'alphaE': dataset[i][7], 'zE': dataset[i][8],
                       'thrust': dataset[i][10], 'betaCorr': dataset[i][11], 'rotCorr': dataset[i][12],
                       })
        data = pd.DataFrame(sheets[i])  # Create a DataFrame
        if not load:
            data.to_excel(writer, sheet_name="wo - Drone " + str(i), index=False)  # Write data to the sheet
        else:
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                data.to_excel(writer, sheet_name="w - Drone " + str(i), index=False)  # Create the sheets or overwrite them

    if not load:  # Close the writer if it was created
        writer.close()




def saveData(quadcopters, dataset):
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
swarm = Swarm(sim,4)
dataset = [[[] for x in range(13)] for y in range(swarm.size)] #Create the dataset with 13 positions for each drone

while (True):
    try:
        if swarm.quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
            sysCall_cleanup(sim,dataset,4,swarm.load,swarm.filename,swarm.trajectoryType,True)
            break
    
        swarm.update_simulation()
        saveData(swarm.quadcopters,dataset); # Save the step data to the dataset

        client.step();

    except: #If an exception occurs, end the program
        sysCall_cleanup(sim,dataset,4,swarm.load,swarm.filename,swarm.trajectoryType,True)
        traceback.print_exc() # Print the exception message
        break

