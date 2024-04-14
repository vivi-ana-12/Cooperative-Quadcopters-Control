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
    update_arrays(actualInputTrajectory,results[sheets[0]].iloc[iteration,1:4].values.tolist())
    

model = tf.keras.models.load_model(".\\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5") 

results, sheets = readAllSheets("..\\DataBase\\Test Results\\Test_Trajectory"+str(11)+"_Results.xlsx")

num_cols = 3
initial_values = [0] * num_cols

delayedTrajectory = pd.DataFrame(np.zeros((17, 3)), columns=['x', 'y', 'z'])
actualInputTrajectory = pd.DataFrame(np.zeros((17, 3)), columns=['x', 'y', 'z'])
predictions = pd.DataFrame()
input_features = pd.DataFrame()
actual = [[],[],[]]
previous =  [[],[],[]]              


for iteration in range (len(results[sheets[0]])):
    if iteration < 17 :
        update_arrays(delayedTrajectory,results[sheets[0]].iloc[iteration,1:4].values.tolist())
    if iteration > 0 and iteration < 18 :
        update_arrays(actualInputTrajectory,results[sheets[0]].iloc[iteration,4:7].values.tolist())

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
        update_arrays(actualInputTrajectory,results[sheets[0]].iloc[iteration,4:7].values.tolist())


# Graficar cada columna por separado
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)  # Subplot para la primera columna
plt.plot(predictions.index, predictions.iloc[:,0])
plt.title('Columna X')

plt.subplot(3, 1, 2)  # Subplot para la segunda columna
plt.plot(predictions.index, predictions.iloc[:,1])
plt.title('Columna Y')

plt.subplot(3, 1, 3)  # Subplot para la tercera columna
plt.plot(predictions.index, predictions.iloc[:,2])
plt.title('Columna Z')

plt.tight_layout()
plt.show()
        
