import tensorflow as tf
import pandas as pd
import numpy as np
import os.path
import matplotlib.pyplot as plt



def readAllSheets(filename):
    if not os.path.isfile(filename):
        return None
    
    xls = pd.ExcelFile(filename)
    sheets = xls.sheet_names
    results = {}
    for sheet in sheets:
        results[sheet] = xls.parse(sheet)
        
    xls.close()
    
    return results, sheets

def update_arrays(array, newData):
    array.iloc[1:, :] = array.iloc[:-1,:].values
    array.loc[0] = newData
    
    return array

def predict_unloaded_behavior(model,actualInputTrajectory,delayedTrajectory,predictions,iteration,input_features):
    input_features = np.concatenate((delayedTrajectory,delayedTrajectory), axis=1)
    print(input_features)
    actualPrediction = pd.DataFrame(model.predict([input_features],verbose=0))
    predictions = pd.concat([predictions,actualPrediction], ignore_index=True,sort = False) #The prediction is made

    update_arrays(delayedTrajectory,actualPrediction.values.tolist())
    update_arrays(actualInputTrajectory,results[sheets[3]].iloc[iteration,1:4].values.tolist())
    

# model = tf.keras.models.load_model(".\\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5") 
model = tf.keras.models.load_model(r"C:\Users\vivia\OneDrive\Graduation Project\CoppeliaSim Files\Simulation Environment\Cooperative Quadcopters Control\Models\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5") 
results, sheets = readAllSheets("..\\DataBase\\Test Trajectories\\TestTrajectory_"+str(4)+"_Results.xlsx")
#C:\Users\vivia\OneDrive\Graduation Project\CoppeliaSim Files\Simulation Environment\Failed but useful attempts\Models\8 delays\Tests
# predictions, pSheets = readAllSheets(r"C:\Users\vivia\OneDrive\Graduation Project\CoppeliaSim Files\Simulation Environment\Failed but useful attempts\Models\6 delays\6 delays, 64-32-16-8 Decimated\64-32-16-8 Decimated - Test Data.xlsx")
num_cols = 3
initial_values = [0] * num_cols

delayedTrajectory = pd.DataFrame(np.zeros((17, 3)), columns=['x', 'y', 'z'])
actualInputTrajectory = pd.DataFrame(np.zeros((17, 3)), columns=['x', 'y', 'z'])
predictions = pd.DataFrame()
input_features = pd.DataFrame()
actual = [[],[],[]]
previous =  [[],[],[]]              

if True:
    for iteration in range (len(results[sheets[0]])):
        if iteration < 17 :
            update_arrays(delayedTrajectory,(results[sheets[0]].iloc[iteration,1:4].values).tolist())
        if iteration > 0 and iteration < 18 :
            update_arrays(actualInputTrajectory,(results[sheets[0]].iloc[iteration,4:7].values).tolist())
    
        # if iteration > 0 and iteration < 18 : #Saves the value of the target position of the drone for the prediction
        #     actual[0].append(results[sheets[0]].iloc[iteration,4])
        #     actual[1].append(results[sheets[0]].iloc[iteration,5])
        #     actual[2].append(results[sheets[0]].iloc[iteration,6])
            
        # if iteration < 17: #Saves the value of the current position of the drone for the prediction
        #     previous[0].append(results[sheets[0]].iloc[iteration,1])
        #     previous[1].append(results[sheets[0]].iloc[iteration,2])
        #     previous[2].append(results[sheets[0]].iloc[iteration,3])
                            
        if iteration > 17:
            # predict_unloaded_behavior(model,actualInputTrajectory,delayedTrajectory,predictions,iteration,input_features)
            concatenated = pd.concat([actualInputTrajectory['x'],actualInputTrajectory['y'], actualInputTrajectory['z'],
                                      delayedTrajectory['x'],delayedTrajectory['y'],delayedTrajectory['z']])
    
                    
            input_features = np.expand_dims(concatenated.values.flatten(), axis=0)
            
            actualPrediction = pd.DataFrame(model.predict(input_features, verbose=0))
    
            predictions = pd.concat([predictions,actualPrediction], ignore_index=True,sort = False) #The prediction is made
    
            # actualInputs = actual[0]+actual[1]+actual[2]+previous[0]+previous[1]+previous[2] #The input vector is set
            # concatenated = pd.DataFrame(actualInputs)
            # input_features = np.expand_dims(concatenated.values.flatten(), axis=0)       
            # actualPrediction = pd.DataFrame(model.predict(input_features, verbose=0))
            # predictions = pd.concat([predictions,actualPrediction], ignore_index=True,sort = False) #The prediction is made
    
    
            # #New data is added to the dataset. Do not use -for- to save execution time
            # previous[0] = [predictions.iloc[-1][0]]+ previous[0][:-1] 
            # previous[1] = [predictions.iloc[-1][1]]+ previous[1][:-1]
            # previous[2] = [predictions.iloc[-1][2]]+ previous[2][:-1]
            
            # actual[0] = [results[sheets[0]].iloc[iteration,1]] + actual[0][:-1]
            # actual[1] = [results[sheets[0]].iloc[iteration,2]] + actual[1][:-1]
            # actual[2] = [results[sheets[0]].iloc[iteration,3]] + actual[2][:-1]
                            
            update_arrays(delayedTrajectory,actualPrediction.iloc[-1,:].values.tolist())
            update_arrays(actualInputTrajectory,(results[sheets[0]].iloc[iteration,4:7].values).tolist())
    
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.titlesize": 17,
    'figure.figsize': (18.5, 9.5),
    "font.size": 15
})

# Graficar cada columna por separado
delay = 16
time = results[sheets[0]].iloc[:,0]
fig, ax = plt.subplots(3,1,sharex = True)
ax[0].set_title('Trayectoria de Prueba Modelo 12-9-3')

ax[0].grid(True, linestyle='--')

# ax[0].set_title("x")
# ax[0].plot(time[delay+2:],predictions[pSheets[4]].iloc[:-8,0],label='Predicción')
ax[0].plot(time[delay+2:],predictions.iloc[:,0],label='Predicción')
ax[0].plot(time,results[sheets[0]].iloc[:,1],label='Objetivo')
ax[0].plot(time,results[sheets[0]].iloc[:,4],label='Entrada') 

ax[0].legend(bbox_to_anchor=(1,1))
# plt.tight_layout()

# ax[1].set_title("y")
ax[1].grid(True, linestyle='--')
# ax[1].plot(time[delay+2:],predictions[pSheets[4]].iloc[:-8,1])
ax[1].plot(time[delay+2:],predictions.iloc[:,1])
ax[1].plot(time,results[sheets[0]].iloc[:,2])
ax[1].plot(time,results[sheets[0]].iloc[:,5])

# ax[2].set_title("z")
ax[2].grid(True, linestyle='--')
# ax[2].plot(time[delay+2:],predictions[pSheets[4]].iloc[:-8,2])
ax[2].plot(time[delay+2:],predictions.iloc[:,2])
ax[2].plot(time,results[sheets[0]].iloc[:,3])
ax[2].plot(time,results[sheets[0]].iloc[:,6])

ax[2].set_xlabel('t(s)')
ax[0].set_ylabel('x(m)')
ax[1].set_ylabel('y(m)')
ax[2].set_ylabel('z(m)')
ax[0].set_aspect(0.4 / ax[0].get_data_ratio())
ax[1].set_aspect(0.4 / ax[1].get_data_ratio())
ax[2].set_aspect(0.4 / ax[2].get_data_ratio())

plt.tight_layout()
plt.show()

plt.savefig(".\\model-12-9-3_test.pdf")
plt.close()
        
