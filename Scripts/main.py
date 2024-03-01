from zmqRemoteApi import RemoteAPIClient
import traceback
import pandas as pd


import Quadcopter
import VisualizeGraphs


def sysCall_cleanup(sim,dataset,quadcopters_number,load,filename,trajectoryType,saveGraph): 
    sim.stopSimulation(); # Stop and disconnect communication with CoppeliaSim
    print('Disconnected')
    print('Saving file')
    # saveFile(dataset,load,trajectoryType,trajectoryNumber) # Save simulation data ----------------------------------------> IMPORTANT 
    # VisualizeGraphs.plotData(trajectoryNumber,load,saveGraph,trajectoryType)
    print('Program ended') 

    
def saveFile(dataset,load,trajectoryType,trajectoryNumber):
    
    if trajectoryType:     
        filename = "..\\DataBase\\Test trajectories\\TestTrajectory_"+str(trajectoryNumber)+"_Results.xlsx"    
    else: 
        filename = "..\\DataBase\\Training Trajectories\\TrainingTrajectory_"+str(trajectoryNumber)+"_Results.xlsx"            
            
    sheets = [] 
    
    if not load: 
        writer = pd.ExcelWriter(filename) #Create the document or overwrite it
        
    for i in range (4): # Create a sheet for each drone
        sheets.append({'t':dataset[i][9],
                      'x':dataset[i][0],'y':dataset[i][1],'z':dataset[i][2],
                      'target x':dataset[i][3],'target y':dataset[i][4],'target z':dataset[i][5],
                      'betaE':dataset[i][6],'alphaE':dataset[i][7],'zE':dataset[i][8],
                      'thrust':dataset[i][10],'betaCorr':dataset[i][11],'rotCorr':dataset[i][12],
                      })
        data = pd.DataFrame(sheets[i]) #Create a DataFrame
        if not load: 
            data.to_excel(writer, sheet_name="wo - Drone "+str(i), index=False) #Delete all the content of the document and create the sheets
        else: 
            with pd.ExcelWriter(filename ,engine='openpyxl', mode='a', if_sheet_exists ='replace') as writer:
                data.to_excel(writer, sheet_name="w - Drone "+str(i), index=False)  #Create the sheets or overwrite them 
                
    if not load: #With load it is not necessary to carry out this step
            writer.save()  


def saveData(drone,quadcopters,dataset):
    dataset[drone][0].append(quadcopters[drone].pos[0])         # Save the x position 
    dataset[drone][1].append(quadcopters[drone].pos[1])         # Save the y position 
    dataset[drone][2].append(quadcopters[drone].pos[2])         # Save the x position 
    dataset[drone][3].append(quadcopters[drone].targetPos[0])   # Save the x target position 
    dataset[drone][4].append(quadcopters[drone].targetPos[1])   # Save the y target position 
    dataset[drone][5].append(quadcopters[drone].targetPos[2])   # Save the z target position 
    dataset[drone][6].append(quadcopters[drone].betaE)          # Save the error at x position
    dataset[drone][7].append(quadcopters[drone].alphaE)         # Save the error at y position
    dataset[drone][8].append(quadcopters[drone].e)              # Save the error at z position
    dataset[drone][9].append(quadcopters[drone].t)              # Save the time
    dataset[drone][10].append(quadcopters[drone].thrust)        # Save the thrust
    dataset[drone][11].append(quadcopters[drone].betaCorr)      # Save the betaCorr
    dataset[drone][12].append(quadcopters[drone].rotCorr)       # Save the rotCorr
    
    
# ------------------------------------------------------------------------------------------------------------------
print('Program started\n')
client = RemoteAPIClient(); # Start RemoteApp connection client

sim = client.getObject('sim');  # Retrieve the object handle
client.setStepping(True); # Activate staggered mode
print('Connected\n')

sim.startSimulation(); 
client.step(); # One step in the simulation

quadcopters_number = 4 
firstStep = False # Flag to save the target positions in the first iteration

trajectoryType = True # Training False, Test True -----------------------------------> !
trajectoryNumber = "11" # ------------------------------------------------------------> !
load = True # Indicates if the Coppelia simulation is with Load or without Load ------> !

#Path of the Excel document where the trajectory will be read 
if trajectoryType:     
    filename = "..\\DataBase\\Test trajectories\\TestTrajectories.xlsx"    
else: 
    filename = "..\\DataBase\\Training Trajectories\\TrainingTrajectories.xlsx"    
   
results, sheets = VisualizeGraphs.readAllSheets(filename) 
  
quadcopters = [] # Create the list where the Quadcopter instances will be saved
dataset = [[[] for x in range(13)] for y in range(quadcopters_number)] #Create the dataset with 13 positions for each drone
trajectory = results [trajectoryNumber]

for i in range(quadcopters_number):
    quadcopters.append(Quadcopter.Quadcopter(sim,"Quadcopter["+str(i)+"]")) # Create an instance of the Quadcopter class for each Drone
  

while (True):
    try:
        if quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
            sysCall_cleanup(sim,dataset,quadcopters_number,load,filename,trajectoryType,True)
            break
    
        for i in range(quadcopters_number):  # Performs the control and saves the data of each drone in each step 
            quadcopters[i].get_parameters(); # Call Coppelia data
            quadcopters[i].set_controller(); # Calculate the controler
            quadcopters[i].set_velocities(); # Send controller output data to CoppeliaSim
            saveData(i,quadcopters,dataset); # Save the step data to the dataset
          
        if(firstStep == False): # If it is the first step, save the initial target position in each axis
            initPos = [[[] for x in range(3)] for y in range(quadcopters_number)] 
            for quadcopter in range (quadcopters_number):
                initPos[quadcopter][0] = quadcopters[quadcopter].targetPos[0]
                initPos[quadcopter][1] = quadcopters[quadcopter].targetPos[1]
                initPos[quadcopter][2] = quadcopters[quadcopter].targetPos[2]
            firstStep = True
        
        
        if float(quadcopters[0].t).is_integer(): # Every second the target positions of the x, y, and z axes are updated
              for quadcopter in range (4):
                  quadcopters[quadcopter].targetPos[0] = initPos[quadcopter][0]+trajectory["x"][quadcopters[0].t-1] 
                  quadcopters[quadcopter].targetPos[1] = initPos[quadcopter][1]+trajectory["y"][quadcopters[0].t-1]
                  quadcopters[quadcopter].targetPos[2] = initPos[quadcopter][2]+trajectory["z"][quadcopters[0].t-1]
                  
                  sim.setObjectPosition(quadcopters[quadcopter].targetObj,sim.handle_world,[quadcopters[quadcopter].targetPos[0],quadcopters[quadcopter].targetPos[1],quadcopters[quadcopter].targetPos[2]])

        client.step();

        
    except: #If an exception occurs, end the program
        sysCall_cleanup(sim,dataset,quadcopters_number,load,filename,trajectoryType,True) # Call the exit function
        traceback.print_exc() # Print the exception message
        break
