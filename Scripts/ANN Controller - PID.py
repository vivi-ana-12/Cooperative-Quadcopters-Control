from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import traceback
import numpy as np
import csv
import os

import VisualizeGraphs
import Quadcopter


def sysCall_cleanup(sim,dataset,trajectoryNumber,quadcopters_number,allPredictions,predictionsError,controlError,epoch): 
    sim.stopSimulation() # Stop and disconnect communication with CoppeliaSim
    # print('Disconnected')
    # print('Program ended') 
    # print('Saving file')
    # saveFile(dataset,trajectoryNumber,quadcopters_number,predictionsError,initPos,allPredictions,epoch)# Save a file with the simulation data
    # saveControlError(predictionsError,trajectoryNumber,quadcopters_number,epoch) # Save a file with the error data for K correction.
    # saveK(dataset,trajectoryNumber,quadcopters_number,epoch)
    # print('Ploting data')
    # plotData(trajectoryNumber,quadcopters_number,delay,epoch) # Generates the graphs taking the data from the file
    # plotControlData(trajectoryNumber)

def plotData(trajectoryNumber,quadcopters_number,delay,epoch):
    filename_test2 = ".\\DataBase\\Test Results\\Test 1\\Test1_Trajectory"+str(trajectoryNumber)+"_Results.xlsx"    
    
    filename_Test = ".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_"+str(epoch)+"_Results.xlsx"

    results, sheets = VisualizeGraphs.readAllSheets(filename_Test) # Open and read the file
    results2, sheets2 = VisualizeGraphs.readAllSheets(filename_test2) # Open and read the file

    dataset = [[[] for x in range(19)] for y in range(len(sheets))]
    dataset_test2 = [[[] for x in range(19)] for y in range(len(sheets2))] 

    for sheet in range (len(sheets)): # Fill the dataset with the data from the file
        for variable in range (19): 
            dataset[sheet][variable].extend(results[sheets[sheet]].to_numpy().transpose().tolist()[variable])
            dataset_test2[sheet][variable].extend(results2[sheets2[sheet]].to_numpy().transpose().tolist()[variable])


    variables = ["x(m)","y(m)","z(m)","Error x","Error y","Error z"] #Variable names per row
    
    plt.rcParams.update({  #Style settings
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.titlesize" : 17,
    'figure.figsize' : (18.5,9.5),
    "font.size": 13
    })
    
    plt.close('all')  #Close all open figures

    image = 1 #Initialize the image to print
    
    fig, ax = plt.subplots(len(variables),quadcopters_number,sharex = True)
    
    for variable in range (6): #Rows 
        for drone in range (quadcopters_number): #Columns
            if(variable < 3): # In the first 3 rows print the target, simulation and prediction data for each axis (x,y and z)
                ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+4],"g")
                ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset[drone][variable+7][0:-delay-2],"b",linestyle = '--')
                ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+1],"r")
        
            else: # In the next 3 rows print the error of each axis
                ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset[drone][variable+7][0:-delay-2],"r")
                ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset_test2[drone][variable+7][0:-delay-2],'c',linestyle = '--')
                
            if (image <= quadcopters_number ): # Print the drone number in each column
                ax[variable, drone].set_title(r'Quadcopter '+str(image-1))

            if (image == 4):
                custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2),Line2D([0], [0], color='blue', lw=2,linestyle = ':')]
                ax[variable, drone].legend(custom_lines, ['Simulation', 'Target','Prediction'], bbox_to_anchor = (1.05,1.08)) 
                
            if (image == 16):
                custom_lines = [Line2D([0], [0], color='red', lw=2),Line2D([0], [0], color='c', lw=2,linestyle = ':')]
                ax[variable, drone].legend(custom_lines, ['W Controller', 'WO Controller'], bbox_to_anchor = (1.05,1.08))   
        
            if (drone == 0): # Print the variable in each row
                ax[variable, drone].set_ylabel(variables[variable])
                
            if (variable == len(variables)-1): 
                ax[variable, drone].set_xlabel(r'Time (s)')  # Add the x-axis label of the last row
    
            ax[variable, drone].grid(linestyle = '--', linewidth = 0.5) # Adds grid
            image = image + 1  # Increase to the next position
        
        
    plt.tight_layout() # Adjust the padding between and around subplots.
    plt.show() # Display all open figures.
    
    plt.savefig(".\\Graphs\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_"+str(epoch)+"_W.pdf")

    plt.close('all')  #Close all open figures

def plotControlData(trajectoryNumber):
    variables = ["Kp","EMA x","EMA y","EMA z"] #Variable names per row
    
    fileK = ".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_Ks.csv"
    filePredictionsErrorK = ".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"controlErrors.csv"


    # Read the CSV file into a dataframe
    kValues = pd.read_csv(fileK)
    PredictionsErrorK = pd.read_csv(filePredictionsErrorK)

    plt.rcParams.update({  #Style settings
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.titlesize" : 17,
    'figure.figsize' : (18.5,9.5),
    "font.size": 13
    })

    plt.close('all')  #Close all open figures

    image = 1 #Initialize the image to print

    fig, ax = plt.subplots(len(variables),quadcopters_number,sharex = True)

    for variable in range (len(variables)): #Rows 
        for drone in range (quadcopters_number): #Columns
        
            if variable == 0: 
                ax[variable, drone].plot(kValues.iloc[:,drone+1],"r")
                ax[variable, drone].plot(kValues.iloc[:,drone+5],"g")
                ax[variable, drone].plot(kValues.iloc[:,drone+9],"b")
            else: 
                ax[variable, drone].plot(PredictionsErrorK.iloc[:,(variable-1)*quadcopters_number+drone+1],"b")

            if (image <= quadcopters_number ): # Print the drone number in each column
                ax[variable, drone].set_title('Quadcopter '+str(image-1))
               
            if (drone == 0): # Print the variable in each row
                ax[variable, drone].set_ylabel(variables[variable])
                
            if (variable == len(variables)-1): 
                ax[variable, drone].set_xlabel('Epochs')  # Add the x-axis label of the last row

            ax[variable, drone].grid(linestyle = '--', linewidth = 0.5) # Adds grid
            image = image + 1  # Increase to the next position
        
        
    plt.tight_layout() # Adjust the padding between and around subplots.
    plt.show() # Display all open figures.


    plt.savefig(".\\Graphs\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_"+str(epoch)+"_ControlData.pdf")

    plt.close('all')  #Close all open figures


def saveFile(dataset,trajectoryNumber,quadcopters_number,predictionsError,initPos,allPredictions,epoch):
    sheets = []

    filename = ".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_"+str(epoch)+"_Results.xlsx"

    writer = pd.ExcelWriter(filename) #Create the document or overwrite it
    
    for i in range (quadcopters_number): # Create a sheet for each drone
        Columns = {'t':dataset[i][9],'x':dataset[i][0],'y':dataset[i][1],'z':dataset[i][2],
                      'target x':dataset[i][3],'target y':dataset[i][4],'target z':dataset[i][5],
                      'prediction x': allPredictions[i][0], 'prediction y': allPredictions[i][1],'prediction z': allPredictions[i][2],
                      'predictionE x': predictionsError[i][0], 'predictionE y': predictionsError[i][1],'predictionE z': predictionsError[i][2],
                      'betaE':dataset[i][6],'alphaE':dataset[i][7],'zE':dataset[i][8],
                      'thrust':dataset[i][10],'betaCorr':dataset[i][11],'rotCorr':dataset[i][12]}
        

        Columns.update({'kP x':dataset[i][13],'kP y':dataset[i][14],'kP z':dataset[i][15],
                        'kD x':dataset[i][16],'kD y':dataset[i][17],'kD z':dataset[i][18],
                        'kI x':dataset[i][19],'kI y':dataset[i][20],'kI z':dataset[i][21]})


        sheets.append(Columns)
        
        data = pd.DataFrame.from_dict(sheets[i], orient='index').T
        
        data.to_excel(writer, sheet_name="w - Drone "+str(i), index=False) 
        
    writer.close()  

def saveControlError(predictionsError,trajectoryNumber,quadcopters_number,epoch):
    file = 'PredictionsErrorK.csv'
    
    filename = ".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"controlErrors.csv"


    # Check if the CSV file exists.
    if os.path.isfile(filename):
        openingMode = 'a'  # The opening mode 'a' is used to append data to a file.
    else:
        openingMode = 'w'  # The opening mode 'w' is used to create a new file
    
    # Data to add to the CSV file
    data = [[epoch, predictionsError[0][0], predictionsError[1][0], predictionsError[2][0], predictionsError[3][0],
                    predictionsError[0][1], predictionsError[1][1], predictionsError[2][1], predictionsError[3][1],
                    predictionsError[0][2], predictionsError[1][2], predictionsError[2][2], predictionsError[3][2]]]
    
    # Open the CSV file in the appropriate mode.
    with open(filename, openingMode, newline='') as file:
        writer = csv.writer(file)
        if openingMode == 'w':
            # Write the header of the file if it is new.
            writer.writerow(['Epoch', 'ErrorK x - D0', 'ErrorK x - D1','ErrorK x - D2','ErrorK x - D3'
                                    , 'ErrorK y - D0', 'ErrorK y - D1','ErrorK y - D2','ErrorK y - D3'
                                    , 'ErrorK z - D0', 'ErrorK z - D1','ErrorK z - D2','ErrorK z - D3'])
    
        # Add the new data to the file.
        writer.writerows(data)

def saveK(dataset,trajectoryNumber,quadcopters_number,epoch):

    filename = ".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_Ks.csv"


    # Check if the CSV file exists.
    if os.path.isfile(filename):
        openingMode = 'a'  # The opening mode 'a' is used to append data to a file.
    else:
        openingMode = 'w'  # The opening mode 'w' is used to create a new file
    
    # Data to add to the CSV file
    data = [[epoch, dataset[0][13][-1],dataset[0][14][-1],dataset[0][15][-1]
                  , dataset[0][19][-1],dataset[0][20][-1],dataset[0][21][-1]
                  , dataset[0][16][-1],dataset[0][17][-1],dataset[0][18][-1]
                  , dataset[1][13][-1],dataset[1][14][-1],dataset[1][15][-1]
                  , dataset[1][19][-1],dataset[1][20][-1],dataset[1][21][-1]
                  , dataset[1][16][-1],dataset[1][17][-1],dataset[1][18][-1]
                  , dataset[2][13][-1],dataset[2][14][-1],dataset[2][15][-1]
                  , dataset[2][19][-1],dataset[2][20][-1],dataset[2][21][-1]
                  , dataset[2][16][-1],dataset[2][17][-1],dataset[2][18][-1]
                  , dataset[3][13][-1],dataset[3][14][-1],dataset[3][15][-1]
                  , dataset[3][19][-1],dataset[3][20][-1],dataset[3][21][-1]
                  , dataset[3][16][-1],dataset[3][17][-1],dataset[3][18][-1]]]
    
    # Open the CSV file in the appropriate mode.
    with open(filename, openingMode, newline='') as file:
        writer = csv.writer(file)
        if openingMode == 'w':
            # Write the header of the file if it is new.
            writer.writerow(['Epoch' , 'Kp x - D0', 'Kp y - D0','Kp z - D0'
                                     , 'Ki x - D0', 'Ki y - D0','Ki z - D0'
                                     , 'Kd x - D0', 'Kd y - D0','Kd z - D0'
                                     , 'Kp x - D1', 'Kp y - D1','Kp z - D1'
                                     , 'Ki x - D1', 'Ki y - D1','Ki z - D1'
                                     , 'Kd x - D1', 'Kd y - D1','Kd z - D1'
                                     , 'Kp x - D2', 'Kp y - D2','Kp z - D2'
                                     , 'Ki x - D2', 'Ki y - D2','Ki z - D2'
                                     , 'Kd x - D2', 'Kd y - D2','Kd z - D2'
                                     , 'Kp x - D3', 'Kp y - D3','Kp z - D3'
                                     , 'Ki x - D3', 'Ki y - D3','Ki z - D3'
                                     , 'Kd x - D3', 'Kd y - D3','Kd z - D3'])
    
        # Add the new data to the file.
        writer.writerows(data)
    

def calculateError(dataset,quadcopters_number,allPredictions,predictionsError): 
        
    for quadcopter in range (quadcopters_number):
        predictionsError[quadcopter][0].append(allPredictions[quadcopter][0][-1] - dataset[quadcopter][0][-1])
        predictionsError[quadcopter][1].append(allPredictions[quadcopter][1][-1] - dataset[quadcopter][1][-1])
        predictionsError[quadcopter][2].append(allPredictions[quadcopter][2][-1] - dataset[quadcopter][2][-1])

    return predictionsError



def calculateAllPredictions(dataset,actualPrediction,initPos,quadcopters_number,ownFramePos,allPredictions):
    
    for quadcopter in range (quadcopters_number):
        allPredictions[quadcopter][0].append(initPos[quadcopter][0] + (actualPrediction.iloc[-1][0]-initPos[0][0]-ownFramePos[0]))
        allPredictions[quadcopter][1].append(initPos[quadcopter][1] + (actualPrediction.iloc[-1][1]-initPos[0][1]-ownFramePos[1]))
        allPredictions[quadcopter][2].append(initPos[quadcopter][2] + (actualPrediction.iloc[-1][2]-initPos[0][2]-ownFramePos[2]))
        
    return allPredictions


def calculateControlError(dataset,quadcopters_number,predictionsError,controlError):
    
    for quadcopter in range (quadcopters_number):
        controlError[quadcopter][0].append(dataset[quadcopter][3][-1] - predictionsError[quadcopter][0][-1])
        controlError[quadcopter][1].append(dataset[quadcopter][4][-1] - predictionsError[quadcopter][1][-1])
        controlError[quadcopter][2].append(dataset[quadcopter][5][-1] - predictionsError[quadcopter][2][-1])
    return controlError

def saveData(drone,quadcopters,dataset,iteration,predictionsError,targetTrajectory):
    dataset[drone][0].append(quadcopters[drone].pos[0])         # Save the x position 
    dataset[drone][1].append(quadcopters[drone].pos[1])         # Save the y position 
    dataset[drone][2].append(quadcopters[drone].pos[2])         # Save the x position
    if iteration < ((delay+1)*2)+3:
        dataset[drone][3].append(quadcopters[drone].targetPos[0]) # Save the x target position 
        dataset[drone][4].append(quadcopters[drone].targetPos[1]) # Save the y target position 
        dataset[drone][5].append(quadcopters[drone].targetPos[2]) # Save the z target position 
    else: 
        dataset[drone][3].append(targetTrajectory[drone][0])    # Save the x target position 
        dataset[drone][4].append(targetTrajectory[drone][1])    # Save the y target position 
        dataset[drone][5].append(targetTrajectory[drone][2])    # Save the z target position 
    dataset[drone][6].append(quadcopters[drone].betaE)          # Save the error at x position
    dataset[drone][7].append(quadcopters[drone].alphaE)         # Save the error at y position
    dataset[drone][8].append(quadcopters[drone].e)              # Save the error at z position
    dataset[drone][9].append(quadcopters[drone].t)              # Save the time
    dataset[drone][10].append(quadcopters[drone].thrust)        # Save the thrust
    dataset[drone][11].append(quadcopters[drone].betaCorr)      # Save the betaCorr
    dataset[drone][12].append(quadcopters[drone].rotCorr)       # Save the rotCorr
    dataset[drone][13].append(quadcopters[drone].kP[0])         # Save the kP of the x-axis
    dataset[drone][14].append(quadcopters[drone].kP[1])         # Save the kP of the y-axis
    dataset[drone][15].append(quadcopters[drone].kP[2])         # Save the kP of the z-axis
    dataset[drone][16].append(quadcopters[drone].kD[0])         # Save the kD of the x-axis
    dataset[drone][17].append(quadcopters[drone].kD[1])         # Save the kD of the y-axis
    dataset[drone][18].append(quadcopters[drone].kD[2])         # Save the kD of the z-axis
    dataset[drone][19].append(quadcopters[drone].kI[0])         # Save the kI of the x-axis
    dataset[drone][20].append(quadcopters[drone].kI[1])         # Save the kI of the y-axis
    dataset[drone][21].append(quadcopters[drone].kI[2])         # Save the kI of the z-axis

# -----------------------------------------------------------------------------------------------------------------------------
#time.sleep(3)

kpValues = [[0 for x in range(3)] for y in range(4)]
kdValues = [[0 for x in range(3)] for y in range(4)]  
kiValues = [[0 for x in range(3)] for y in range(4)]  


quadcopters_number = 4 # Number of quadcopters to simulate 
delay = 16 # Number of delays of the prediction model

trajectoryType = True # Training False, Test True -----------------------------------> !
trajectoryNumber = "11" # -----------------------------------------------------------> !

#Path of the Excel document where the trajectory will be read 
if trajectoryType:     
    filename = ".\\DataBase\\Test trajectories\\TestTrajectories.xlsx"    
else: 
    filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories.xlsx"    
   
results, sheets = VisualizeGraphs.readAllSheets(filename) 
trajectory = results [trajectoryNumber]


# PredictionsErrorFile = pd.read_csv(".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_PredictionsError.csv")


model = tf.keras.models.load_model(".\\models\\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5") 
print(model.summary())


predictionsError = [[[] for x in range(3)] for y in range(quadcopters_number)]
controlError = [[[] for x in range(3)] for y in range(quadcopters_number)]
lastTarget = [[0 for x in range(3)] for y in range(quadcopters_number)]


for epoch in range (1):
    
    if epoch != 0:
        KsFile = pd.read_csv(".\\DataBase\\Test Results\\Test 5 - ANN PID Controller GD\\Trajectory "+str(trajectoryNumber)+"\\Test5_Trajectory"+str(trajectoryNumber)+"_Ks.csv")
        
        for drone in range(4):
            for axis in range(3):
                kpValues[drone][axis] = KsFile.iloc[-1,9*drone+axis+1]
                kiValues[drone][axis] = KsFile.iloc[-1,9*drone+axis+4]
                kdValues[drone][axis] = KsFile.iloc[-1,9*drone+axis+7]
        
        del KsFile


    print("---------------------")
    print("epoch "+str(epoch)+":")
              
    print('Program started\n')
    client = RemoteAPIClient(); # Start RemoteApp connection client
    
    sim = client.getObject('sim');  # Retrieve the object handle
    client.setStepping(True); # Activate staggered mode
    print('Connected\n')
    
    sim.startSimulation(); 
    sim.intparam_speedmodifier = True
    # sim.boolparam_display_enabled  = True
    client.step(); # One step in the simulation
    firstStep = False # Flag to save the target positions in the first iteration
    
      
    quadcopters = [] # Create the list where the Quadcopter instances will be saved
    dataset = [[[] for x in range(22)] for y in range(quadcopters_number)] # Create the dataset with 13 positions for each drone
    initPos = [[[] for x in range(3)] for y in range(quadcopters_number)]  # Create a dataset to store the initial position of each drone
    
    iteration = 0 # Indicates the number of simulation steps
    
    ownFramePos = [] # Stores the own frame position of the drone 0
    # -- Variables to make the prediction --
    actual = [[],[],[]]
    previous =  [[],[],[]]              
    predictions = pd.DataFrame()
    actualPrediction = pd.DataFrame()
    allPredictions = [[[] for x in range(3)] for y in range(quadcopters_number)]
    
    targetTrajectory = [[[] for x in range(3)] for y in range(quadcopters_number)]  
    
    
    for i in range(quadcopters_number):
        quadcopters.append(Quadcopter.Quadcopter(sim,"Quadcopter["+str(i)+"]")) # Create an instance of the Quadcopter class for each Drone
        quadcopters[i].kP = kpValues[i]
        quadcopters[i].kI = kiValues[i]
        quadcopters[i].kD = kdValues[i]
        
    
    while (True):
        try:
            if quadcopters[0].t >= 240: # If the simulation has already reached 240 sec (4 mins), stop it
    
                sysCall_cleanup(sim,dataset,trajectoryNumber,quadcopters_number,allPredictions,predictionsError,controlError,epoch) # Call the exit function
                break
                
            for quadcopter in range(quadcopters_number):  # Performs the control and saves the data of each drone in each step 
                quadcopters[quadcopter].get_parameters(); # Call Coppelia data
                quadcopters[quadcopter].set_controller(); # Calculate the controler
                quadcopters[quadcopter].set_velocities(); # Send controller output data to CoppeliaSim
                if iteration%2 == 0: #Decimate
                    saveData(quadcopter,quadcopters,dataset,iteration,predictionsError,targetTrajectory); # Save the step data to the dataset
        
            if(firstStep == False): # If it is the first step, save the initial target position in each axis           
                ownFramePos.append(0.738 - quadcopters[0].pos[0])
                ownFramePos.append(-0.755 - quadcopters[0].pos[1])
                ownFramePos.append(1.6505 - quadcopters[0].pos[2])
    
                for quadcopter in range (quadcopters_number): # Save the initial position
                    initPos[quadcopter][0] = quadcopters[quadcopter].targetPos[0]
                    initPos[quadcopter][1] = quadcopters[quadcopter].targetPos[1]
                    initPos[quadcopter][2] = quadcopters[quadcopter].targetPos[2]   
                    
                firstStep = True   
        
            if float(quadcopters[0].t).is_integer(): # Every second the target positions of the x, y, and z axes are updated
                  for quadcopter in range (quadcopters_number):
                      targetTrajectory[quadcopter][0] = initPos[quadcopter][0]+trajectory["x"][quadcopters[0].t-1] 
                      targetTrajectory[quadcopter][1] = initPos[quadcopter][1]+trajectory["y"][quadcopters[1].t-1] 
                      targetTrajectory[quadcopter][2] = initPos[quadcopter][2]+trajectory["z"][quadcopters[2].t-1]
                      
                      if iteration < ((delay+1)*2)+3:
                          quadcopters[quadcopter].targetPos[0] = targetTrajectory[quadcopter][0] 
                          quadcopters[quadcopter].targetPos[1] = targetTrajectory[quadcopter][1]
                          quadcopters[quadcopter].targetPos[2] = targetTrajectory[quadcopter][2]
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
                        
                        actualPrediction = pd.DataFrame(model.predict([actualInputs],verbose=0))
                        predictions = pd.concat([predictions,actualPrediction], ignore_index=True,sort = False) #The prediction is made
            
                        #New data is added to the dataset. Do not use -for- to save execution time
                        previous[0] = [predictions.iloc[-1][0]]+ previous[0][:-1] 
                        previous[1] = [predictions.iloc[-1][1]]+ previous[1][:-1]
                        previous[2] = [predictions.iloc[-1][2]]+ previous[2][:-1]
                        
                        actual[0] = [targetTrajectory[0][0] + ownFramePos[0]] + actual[0][:-1]
                        actual[1] = [targetTrajectory[0][1] + ownFramePos[1]] + actual[1][:-1]
                        actual[2] = [targetTrajectory[0][2] + ownFramePos[2]] + actual[2][:-1]
                        
                        allPredictions = calculateAllPredictions(dataset,actualPrediction,initPos,quadcopters_number,ownFramePos,allPredictions)
                        predictionsError = calculateError(dataset,quadcopters_number,allPredictions,predictionsError)
                        controlError = calculateControlError(dataset,quadcopters_number,predictionsError,controlError)
                        
                if iteration > ((delay+1)*2)+2:
                    for quadcopter in range (quadcopters_number):
                        
                        # quadcopters[quadcopter].set_ANN_controller(predictionsError[quadcopter], dataset[quadcopter], np.array(quadcopters[quadcopter].targetPos) - np.array(lastTarget[quadcopter]))

                        quadcopters[quadcopter].set_ANN_controller(predictionsError[quadcopter],dataset[quadcopter])


                        if iteration >= ((delay + 1) * 2) + 8:
                            quadcopters[quadcopter].targetPos[0] = targetTrajectory[quadcopter][0] + quadcopters[quadcopter].kP[0]*(predictionsError[quadcopter][0][-1]) + quadcopters[quadcopter].kI[0]*(predictionsError[quadcopter][0][-1] + predictionsError[quadcopter][0][-2] + predictionsError[quadcopter][0][-3] + predictionsError[quadcopter][0][-4]) + quadcopters[quadcopter].kD[0]*(predictionsError[quadcopter][0][-1] - predictionsError[quadcopter][0][-2])  
                            quadcopters[quadcopter].targetPos[1] = targetTrajectory[quadcopter][1] + quadcopters[quadcopter].kP[1]*(predictionsError[quadcopter][1][-1]) + quadcopters[quadcopter].kI[1]*(predictionsError[quadcopter][1][-1] + predictionsError[quadcopter][1][-2] + predictionsError[quadcopter][1][-3] + predictionsError[quadcopter][1][-4]) + quadcopters[quadcopter].kD[0]*(predictionsError[quadcopter][1][-1] - predictionsError[quadcopter][1][-2])
                            quadcopters[quadcopter].targetPos[2] = targetTrajectory[quadcopter][2] + quadcopters[quadcopter].kP[2]*(predictionsError[quadcopter][2][-1]) + quadcopters[quadcopter].kI[2]*(predictionsError[quadcopter][2][-1] + predictionsError[quadcopter][2][-2] + predictionsError[quadcopter][2][-3] + predictionsError[quadcopter][2][-4]) + quadcopters[quadcopter].kD[0]*(predictionsError[quadcopter][2][-1] - predictionsError[quadcopter][2][-2])                        
                        else: 
                            quadcopters[quadcopter].targetPos[0] = targetTrajectory[quadcopter][0] + quadcopters[quadcopter].kP[0]*(predictionsError[quadcopter][0][-1]) + quadcopters[quadcopter].kD[0]*(predictionsError[quadcopter][0][-1] - predictionsError[quadcopter][0][-2])  
                            quadcopters[quadcopter].targetPos[1] = targetTrajectory[quadcopter][1] + quadcopters[quadcopter].kP[1]*(predictionsError[quadcopter][1][-1]) + quadcopters[quadcopter].kD[0]*(predictionsError[quadcopter][1][-1] - predictionsError[quadcopter][1][-2])
                            quadcopters[quadcopter].targetPos[2] = targetTrajectory[quadcopter][2] + quadcopters[quadcopter].kP[2]*(predictionsError[quadcopter][2][-1]) + quadcopters[quadcopter].kD[0]*(predictionsError[quadcopter][2][-1] - predictionsError[quadcopter][2][-2]) 
                        
                        
                        # lastTarget[quadcopter] = quadcopters[quadcopter].targetPos


                        # if iteration >= ((delay + 1) * 2) + 8:
                        #     for axis in range(3):
                        #         quadcopters[quadcopter].targetPos[axis] = targetTrajectory[quadcopter][axis] + \
                        #                                                         quadcopters[quadcopter].kP[axis] * \
                        #                                                         predictionsError[quadcopter][axis][-1] + \
                        #                                                         quadcopters[quadcopter].kI[axis] * \
                        #                                                         sum(predictionsError[quadcopter][axis][-1:-5:-1]) + \
                        #                                                         quadcopters[quadcopter].kD[axis] * \
                        #                                                         (predictionsError[quadcopter][axis][-1] -
                        #                                                          predictionsError[quadcopter][axis][-2])
                    

                        # else:
                        #     for axis in range(3):
                        #         quadcopters[quadcopter].targetPos[axis] = targetTrajectory[quadcopter][axis] + \
                        #                                                         quadcopters[quadcopter].kP[axis] * \
                        #                                                         predictionsError[quadcopter][axis][-1] + \
                        #                                                         quadcopters[quadcopter].kD[axis] * \
                        #                                                         (predictionsError[quadcopter][axis][-1] -
                        #                                                          predictionsError[quadcopter][axis][-2])



                            
                        

                        sim.setObjectPosition(quadcopters[quadcopter].targetObj,sim.handle_world,[quadcopters[quadcopter].targetPos[0],quadcopters[quadcopter].targetPos[1],quadcopters[quadcopter].targetPos[2]])
                            
            iteration = iteration + 1
            client.step();
    
        except: #If an exception occurs, end the program
            sysCall_cleanup(sim,dataset,trajectoryNumber,quadcopters_number,allPredictions,predictionsError,controlError,epoch) # Call the exit function
            traceback.print_exc() # Print the exception message
            break

