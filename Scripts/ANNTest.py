# The following code performs the simulation of the drones and the ANN prediction simultaneously.

# To evaluate the network's prediction accuracy, simulate eight drones simultaneously in the "Neural Network - Test" scene.
# the code compares the real-time positions of the unloaded drones with their predicted positions.

# To save the behavior without the ANN handler of the loaded drones, run with the "4 Dynamic Strings with Load - Python Communication" scene.

# In both cases the simulated data and the network prediction will be saved.

from zmqRemoteApi import RemoteAPIClient
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import traceback

import VisualizeGraphs
import Quadcopter


def sysCall_cleanup(sim,dataset,quadcopters_number,initPos,predictions,ownFramePos): 
    sim.stopSimulation() # Stop and disconnect communication with CoppeliaSim
    print('Disconnected')
    print('Program ended') 
    allPredictions = calculateAllPredictions(dataset,predictions,initPos,quadcopters_number,ownFramePos) # Create the prediction for each drone based on the prediction of drone 0
    predictionsError = calculateError(dataset,predictions,initPos,quadcopters_number,allPredictions)     # Calculate the error of each prediction
    print('Saving file')
    saveFile(dataset,trajectoryNumber,quadcopters_number,predictionsError,initPos,allPredictions)        # Save a file with the simulation data
    plotData(trajectoryNumber,quadcopters_number,delay) # Generates the graphs taking the data from the file
     
    
def plotData(trajectoryNumber,quadcopters_number,delay):
    filename = ".\\DataBase\\Test Results\\Test 1\\Test1_Trajectory"+str(trajectoryNumber)+"_Results.xlsx" 

    results, sheets = VisualizeGraphs.readAllSheets(filename) # Open and read the file
    
    dataset = [[[] for x in range(19)] for y in range(len(sheets))] 
    
    
    for sheet in range (len(sheets)): # Fill the dataset with the data from the file
        for variable in range (19): 
            dataset[sheet][variable].extend(results[sheets[sheet]].to_numpy().transpose().tolist()[variable])


    variables = ["x(m)","y(m)","z(m)","Error x","Error y","Error z"] #Variable names per row
    
    plt.rcParams.update({  #Style settings
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.titlesize" : 17,
    'figure.figsize' : (18.5,9.5),
    "font.size": 13
    })
    
    plt.close('all')  #Close all open figures
    
    drones = quadcopters_number
    
    for graph in range (int(drones/4)):
        quadcopters_number = 4
        
        image = 1 #Initialize the image to print
        
        fig, ax = plt.subplots(len(variables),quadcopters_number,sharex = True)
        
        for variable in range (6): #Rows 
            for drone in range (quadcopters_number): #Columns
                if(variable < 3): # In the first 3 rows print the target, simulation and prediction data for each axis (x,y and z)
                    if drones == 8 and graph == 0: 
                        ax[variable, drone].plot(dataset[drone+4][0],dataset[drone+4][variable+4],"g")
                        ax[variable, drone].plot(dataset[drone+4][0][delay+2:],dataset[drone+4][variable+7][0:-delay-2],"b")
                        ax[variable, drone].plot(dataset[drone+4][0],dataset[drone+4][variable+1],"r")
                    else: 
                        ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+4],"g")
                        ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset[drone][variable+7][0:-delay-2],"b")
                        ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+1],"r")
                
                else: # In the next 3 rows print the error of each axis
                    if drones == 8 and graph == 0: 
                        ax[variable, drone].plot(dataset[drone+4][0][delay+2:],dataset[drone+4][variable+7][0:-delay-2],"b")
                    else: 
                        ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset[drone][variable+7][0:-delay-2],"b")

                        
                if (image <= quadcopters_number ): # Print the drone number in each column
                    if drones == 8 and graph == 0: 
                        ax[variable, drone].set_title(r'Quadcopter '+str(image+3))
                    else: 
                        ax[variable, drone].set_title(r'Quadcopter '+str(image-1))

                if (image == 4):
                    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2),Line2D([0], [0], color='blue', lw=2,linestyle = ':')]
                    ax[variable, drone].legend(custom_lines, ['Simulation', 'Target','Prediction'], bbox_to_anchor = (1.05,1.08))   
            
                if (drone == 0): # Print the variable in each row
                    ax[variable, drone].set_ylabel(variables[variable])
                    
                if (variable == len(variables)-1): 
                    ax[variable, drone].set_xlabel(r'Time (s)')  # Add the x-axis label of the last row
        
                ax[variable, drone].grid(linestyle = '--', linewidth = 0.5) # Adds grid
                image = image + 1  # Increase to the next position
            
            
        plt.tight_layout() # Adjust the padding between and around subplots.
        plt.show() # Display all open figures.
        
        if drones == 8: # Save the file 
            if graph == 0: 
                plt.savefig(".\\Graphs\\Test Results\\Test 2\\Test2_Trajectory"+str(trajectoryNumber)+"W.pdf")
            else: 
                plt.savefig(".\\Graphs\\Test Results\\Test 2\\Test2_Trajectory"+str(trajectoryNumber)+"WO.pdf")
        else: 
            plt.savefig(".\\Graphs\\Test Results\\Test 1\\Test1_Trajectory"+str(trajectoryNumber)+"WO.pdf")

    
    
def saveFile(dataset,trajectoryNumber,quadcopters_number,predictionsError,initPos,allPredictions):
    sheets = []
    filename = ".\\DataBase\\Test Results\\Test 1\\Test1_Trajectory"+str(trajectoryNumber)+"_Results.xlsx"    
    writer = pd.ExcelWriter(filename) #Create the document or overwrite it
    
    for i in range (quadcopters_number): # Create a sheet for each drone
        sheets.append({'t':dataset[i][9],
                      'x':dataset[i][0],'y':dataset[i][1],'z':dataset[i][2],
                      'target x':dataset[i][3],'target y':dataset[i][4],'target z':dataset[i][5],
                      'prediction x': allPredictions[i][0].tolist(), 'prediction y': allPredictions[i][1].tolist(),'prediction z': allPredictions[i][2].tolist(),
                      'predictionE x': predictionsError[i][0].tolist(), 'predictionE y': predictionsError[i][1].tolist(),'predictionE z': predictionsError[i][2].tolist(),
                      'betaE':dataset[i][6],'alphaE':dataset[i][7],'zE':dataset[i][8],
                      'thrust':dataset[i][10],'betaCorr':dataset[i][11],'rotCorr':dataset[i][12],
                     })
        
        data = pd.DataFrame.from_dict(sheets[i], orient='index').T
        
        if i < 4 : 
            data.to_excel(writer, sheet_name="wo - Drone "+str(i), index=False) 
        else: 
            data.to_excel(writer, sheet_name="w - Drone "+str(i), index=False) 
        
    writer.save()  

def calculateError(dataset,predictions,initPos,quadcopters_number,allPredictions): 
    
    predictionsError = [[[] for x in range(3)] for y in range(quadcopters_number)]  
    
    for quadcopter in range (quadcopters_number):
        for ax in range (3): 
            predictionsError[quadcopter][ax] = allPredictions[quadcopter][ax] - dataset[quadcopter][ax][delay+2:]

    return predictionsError

def calculateAllPredictions(dataset,predictions,initPos,quadcopters_number,ownFramePos):
    
    allPredictions = [[[] for x in range(3)] for y in range(quadcopters_number)]  

    for quadcopter in range (quadcopters_number):
        allPredictions[quadcopter][0] = initPos[quadcopter][0] + (predictions.iloc[:,0]-initPos[0][0]-ownFramePos[0])
        allPredictions[quadcopter][1] = initPos[quadcopter][1] + (predictions.iloc[:,1]-initPos[0][1]-ownFramePos[1])
        allPredictions[quadcopter][2] = initPos[quadcopter][2] + (predictions.iloc[:,2]-initPos[0][2]-ownFramePos[2])
            
    return allPredictions
    
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
    
# -----------------------------------------------------------------------------------------------------------------------------
print('Program started\n')
client = RemoteAPIClient(); # Start RemoteApp connection client

sim = client.getObject('sim');  # Retrieve the object handle
client.setStepping(True); # Activate staggered mode
print('Connected\n')

sim.startSimulation(); 
client.step(); # One step in the simulation

quadcopters_number = 4 # Number of quadcopters to simulate 
firstStep = False # Flag to save the target positions in the first iteration

trajectoryType = True # Training False, Test True -----------------------------------> !
trajectoryNumber = "11" # ------------------------------------------------------------> !

#Path of the Excel document where the trajectory will be read 
if trajectoryType:     
    filename = ".\\DataBase\\Test trajectories\\TestTrajectories.xlsx"    
else: 
    filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories.xlsx"    
   
results, sheets = VisualizeGraphs.readAllSheets(filename) 
trajectory = results [trajectoryNumber]

  
quadcopters = [] # Create the list where the Quadcopter instances will be saved
dataset = [[[] for x in range(13)] for y in range(quadcopters_number)] # Create the dataset with 13 positions for each drone
initPos = [[[] for x in range(3)] for y in range(quadcopters_number)]  # Create a dataset to store the initial position of each drone

delay = 16 # Number of delays of the prediction model
iteration = 0 # Indicates the number of simulation steps

ownFramePos = [] # Stores the own frame position of the drone 0
# -- Variables to make the prediction --
actual = [[],[],[]]
previous =  [[],[],[]]              
predictions = pd.DataFrame()
actualPrediction = pd.DataFrame()
allPredictions = [[[] for x in range(3)] for y in range(quadcopters_number)]
predictionsError = [[[] for x in range(3)] for y in range(quadcopters_number)]  
targetTrajectory = [[[] for x in range(3)] for y in range(quadcopters_number)]  


model = tf.keras.models.load_model(".\\models\\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5") 
print(model.summary())


for i in range(quadcopters_number):
    quadcopters.append(Quadcopter.Quadcopter(sim,"Quadcopter["+str(i)+"]")) # Create an instance of the Quadcopter class for each Drone
 
while (True):
    try:
        if quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
            sysCall_cleanup(sim,dataset,quadcopters_number,initPos,predictions,ownFramePos)
            break
            
        for i in range(quadcopters_number):  # Performs the control and saves the data of each drone in each step 
            quadcopters[i].get_parameters(); # Call Coppelia data
            quadcopters[i].set_controller(); # Calculate the controler
            quadcopters[i].set_velocities(); # Send controller output data to CoppeliaSim
            if iteration%2 == 0: #Decimate
                saveData(i,quadcopters,dataset); # Save the step data to the dataset
            
          
        if(firstStep == False): # If it is the first step, save the initial target position in each axis           
            ownFramePos.append(0.738 - quadcopters[0].pos[0])
            ownFramePos.append(-0.755 - quadcopters[0].pos[1])
            ownFramePos.append(1.6505 - quadcopters[0].pos[2])

            for quadcopter in range (quadcopters_number):
                initPos[quadcopter][0] = quadcopters[quadcopter].targetPos[0]
                initPos[quadcopter][1] = quadcopters[quadcopter].targetPos[1]
                initPos[quadcopter][2] = quadcopters[quadcopter].targetPos[2]    
                
            firstStep = True
        
        if float(quadcopters[0].t).is_integer(): # Every second the target positions of the x, y, and z axes are updated
              for quadcopter in range (quadcopters_number):
                  quadcopters[quadcopter].targetPos[0] = initPos[quadcopter][0]+trajectory["x"][quadcopters[0].t-1] 
                  quadcopters[quadcopter].targetPos[1] = initPos[quadcopter][1]+trajectory["y"][quadcopters[0].t-1]
                  quadcopters[quadcopter].targetPos[2] = initPos[quadcopter][2]+trajectory["z"][quadcopters[0].t-1]
                  
                  sim.setObjectPosition(quadcopters[quadcopter].targetObj,sim.handle_world,[quadcopters[quadcopter].targetPos[0],quadcopters[quadcopter].targetPos[1],quadcopters[quadcopter].targetPos[2]])

        #------------------------------------------------ Prediction -------------------------------
        if iteration%2 == 0: #Decimate
            if iteration <= (delay+1)*2: #In the first iterations this is done since there are not enough values to make the prediction
                if iteration <= (delay+1)*2 and iteration != 0: #Saves the value of the target position of the drone for the prediction
                    actual[0].append(quadcopters[0].targetPos[0]+ownFramePos[0])
                    actual[1].append(quadcopters[0].targetPos[1]+ownFramePos[1])
                    actual[2].append(quadcopters[0].targetPos[2]+ownFramePos[2])
                    
                if iteration < (delay+1)*2: #Saves the value of the current position of the drone for the prediction
                    previous[0].append(quadcopters[0].pos[0]+ownFramePos[0])
                    previous[1].append(quadcopters[0].pos[1]+ownFramePos[1])
                    previous[2].append(quadcopters[0].pos[2]+ownFramePos[2])
            else: #When there are enough values, predictions are made
                    actualInputs = actual[0]+actual[1]+actual[2]+previous[0]+previous[1]+previous[2] #The input vector is set
                
                    predictions = pd.concat([predictions, pd.DataFrame(model.predict([actualInputs],verbose=0))], ignore_index=True,sort = False) #The prediction is made
        
                    #New data is added to the dataset. Do not use -for- to save execution time
                    previous[0] = [predictions.iloc[-1][0]]+ previous[0][:-1] 
                    previous[1] = [predictions.iloc[-1][1]]+ previous[1][:-1]
                    previous[2] = [predictions.iloc[-1][2]]+ previous[2][:-1]
                    
                    actual[0] = [quadcopters[0].targetPos[0]+ownFramePos[0]] + actual[0][:-1]
                    actual[1] = [quadcopters[0].targetPos[1]+ownFramePos[1]] + actual[1][:-1]
                    actual[2] = [quadcopters[0].targetPos[2]+ownFramePos[2]] + actual[2][:-1]
                    

        iteration = iteration + 1
        client.step();

    except: #If an exception occurs, end the program
        sysCall_cleanup(sim,dataset,quadcopters_number,initPos,predictions,ownFramePos) # Call the exit function
        traceback.print_exc() # Print the exception message
        break
