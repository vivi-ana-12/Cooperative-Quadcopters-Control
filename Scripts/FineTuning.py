import os
from os import mkdir
import pandas as pd
import matplotlib.pyplot as plt
import time 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import numpy

def readAllSheets(filename):
    if not os.path.isfile(filename):
        return None
    
    xls = pd.ExcelFile(filename)
    sheets = xls.sheet_names
    results = {}
    for sheet in sheets[0:4]:
        results[sheet] = xls.parse(sheet)
        
    xls.close()
    
    return results, sheets

def createDataSet(delay): 
    inputs = []
    outputs = []
    
    start = time.time()
    
    for trajectory in range (1,131): 
        print(trajectory)
        results, sheets = readAllSheets("..\\DataBase\\Training Trajectories\\TrainingTrajectory_"+str(trajectory)+"_Results.xlsx")
        # for drone in range (4):  
        drone = random.randrange(0, 4, 1)
        actualState = delay+2
        results[sheets[drone]] = results[sheets[drone]].query('index%2 != 0')

        while actualState < len(results[sheets[drone]].iloc[:,0]):
           actualInputs = []
    
           actualInputs.extend(results[sheets[drone]].iloc[actualState-delay-1:actualState,:].loc[:,"target x"].iloc[::-1].to_numpy().flatten())
           actualInputs.extend(results[sheets[drone]].iloc[actualState-delay-1:actualState,:].loc[:,"target y"].iloc[::-1].to_numpy().flatten())
           actualInputs.extend(results[sheets[drone]].iloc[actualState-delay-1:actualState,:].loc[:,"target z"].iloc[::-1].to_numpy().flatten())
    
           # Feedback
           actualInputs.extend(results[sheets[drone]].iloc[actualState-delay-2:actualState-1,:].loc[:,"x"].iloc[::-1].to_numpy().flatten())
           actualInputs.extend(results[sheets[drone]].iloc[actualState-delay-2:actualState-1,:].loc[:,"y"].iloc[::-1].to_numpy().flatten())
           actualInputs.extend(results[sheets[drone]].iloc[actualState-delay-2:actualState-1,:].loc[:,"z"].iloc[::-1].to_numpy().flatten())
    
    
           inputs.append(actualInputs)
    
           outputs.append(results[sheets[drone]].iloc[actualState-1,:].iloc[1:4].to_numpy().flatten())
    
           actualState = actualState + 1
    
    end = time.time()
    print (end-start)
    
    
    #----------------------------------------- Saving Data ------------------------------------------------
    
    pd.DataFrame(inputs).to_csv('..\\models\\'+str(delay)+' delays\\inputsTrainingDecimated100.csv', sep=';')
    pd.DataFrame(outputs).to_csv('..\\models\\'+str(delay)+' delays\\outputsTrainingDecimated100.csv', sep=';')

    return pd.DataFrame(inputs),pd.DataFrame(outputs)
 

def readDataSet(delay):
    # inputs_training = pd.read_csv('.\\Models\\'+str(delay)+' delays\\inputsTraining'+str(delay)+'.csv', sep=';').iloc[:,1:]
    # outputs_training = pd.read_csv('.\\Models\\'+str(delay)+' delays\\outputsTraining'+str(delay)+'.csv', sep=';').iloc[:,1:]
    
    inputs_training = pd.read_csv('..\\models\\'+str(delay)+' delays\\inputsTrainingDecimated100.csv', sep=';').iloc[:,1:]
    outputs_training = pd.read_csv('..\\models\\'+str(delay)+' delays\\outputsTrainingDecimated100.csv', sep=';').iloc[:,1:]

    return inputs_training,outputs_training


delay = 8
modelPath = "64-32-16-8 130"  

model = tf.keras.models.load_model("..\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath +"\\ANN_"+ modelPath +"_"+str(delay)+"delays.h5")


opt = tf.keras.optimizers.Adam(learning_rate=0.00011)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

# inputs_train, outputs_train = createDataSet(delay)
inputs_train, outputs_train = readDataSet(delay)


history_fine = model.fit(inputs_train, outputs_train, validation_split=0.2, epochs=300, batch_size=512)


# loss = pd.read_csv(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\loss.csv", sep=';').iloc[:,1].values.tolist()
# val_loss = pd.read_csv(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\val_loss.csv", sep=';').iloc[:,1].values.tolist()

# loss = history_fine.history['loss']
# val_loss += history_fine.history['val_loss']

# initial_epochs = 300 

# if max(loss) < max(val_loss):
#     maximum = max(val_loss)
# else: 
#     maximum = max(loss)
    
# if min(loss) < min(val_loss):
#     minimum = max(val_loss)
# else: 
#     minimum = max(loss)
    
# plt.figure(figsize=(8, 8))

# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.ylim([-0.00001, maximum])
# plt.plot([initial_epochs-1,initial_epochs-1],
#          plt.ylim(), label='Start Fine Tuning')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.xlabel('epoch')
# plt.show()

modelPath = "64-32-16-8 130 FineTuning - 1"  
mkdir("..\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath)

# plt.savefig(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\"+str(delay)+" delays, "+modelPath+".pdf")
# plt.savefig(".\\models\\"+str(delay)+" delays\\Models Loss - Graphs\\"+str(delay)+" delays, "+modelPath+".jpg")

# plt.close()


#--------------------------------------  Save model -----------------------------------------------



model.save("..\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\ANN_"+modelPath+"_"+str(delay)+"delays.h5")

# pd.DataFrame(loss).to_csv(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\loss.csv", sep=';')
# pd.DataFrame(val_loss).to_csv(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\val_loss.csv", sep=';')



