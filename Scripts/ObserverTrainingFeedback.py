import os
from os import mkdir
import pandas as pd
import matplotlib.pyplot as plt
import time 
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random


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


def SaveData(dataArchitechtures,delay): 
    #print('x:'+str(len(x))+'y:'+str(len(y))+'z:'+str(len(z)))

    filename = '.\\Models\\'+str(delay)+' delays\\'+str(delay)+' delays - Training Data'
    
    sheets = {'Architechture':dataArchitechtures.loc[:,'Architechture'],'Mean_sq':dataArchitechtures.loc[:,'Mean_sq'],'Loss':dataArchitechtures.loc[:,'Loss']}
    data = pd.DataFrame(sheets)
    
    with pd.ExcelWriter(filename+'.xlsx',engine='openpyxl', mode='a', if_sheet_exists ='overlay') as writer:
            data.to_excel(writer, sheet_name="Training Data big- Models", index=False)
            
    
def createDataSet(delay): 
    inputs = []
    outputs = []
    
    start = time.time()
    
    for trajectory in range (1,131): 
        print(trajectory)
        results, sheets = readAllSheets(".\\DataBase\\Training Trajectories\\TrainingTrajectory_"+str(trajectory)+"_Results.xlsx")
        # for drone in range (4):  
        drone = random.randrange(0, 4, 1)
        actualState = delay+2
        results[sheets[drone]] = results[sheets[drone]].query('index%2 != 0')
        
        # results[sheets[drone]].loc[:,"target x"] =  results[sheets[drone]].loc[:,"target x"] - results[sheets[drone]].loc[1,"target x"]
        # results[sheets[drone]].loc[:,"target y"] =  results[sheets[drone]].loc[:,"target y"] - results[sheets[drone]].loc[1,"target y"]
        # results[sheets[drone]].loc[:,"x"] =  results[sheets[drone]].loc[:,"x"] - results[sheets[drone]].loc[1,"x"]
        # results[sheets[drone]].loc[:,"y"] =  results[sheets[drone]].loc[:,"y"] - results[sheets[drone]].loc[1,"y"]


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
    
    pd.DataFrame(inputs).to_csv('.\\models\\'+str(delay)+' delays\\inputsTrainingDecimated100.csv', sep=';')
    pd.DataFrame(outputs).to_csv('.\\models\\'+str(delay)+' delays\\outputsTrainingDecimated100.csv', sep=';')


    return pd.DataFrame(inputs),pd.DataFrame(outputs)
 


def readDataSet(delay):
    # inputs_training = pd.read_csv('.\\Models\\'+str(delay)+' delays\\inputsTraining'+str(delay)+'.csv', sep=';').iloc[:,1:]
    # outputs_training = pd.read_csv('.\\Models\\'+str(delay)+' delays\\outputsTraining'+str(delay)+'.csv', sep=';').iloc[:,1:]
    
    inputs_training = pd.read_csv('.\\models\\'+str(delay)+' delays\\inputsTrainingDecimated131.csv', sep=';').iloc[:,1:]
    outputs_training = pd.read_csv('.\\models\\'+str(delay)+' delays\\outputsTrainingDecimated131.csv', sep=';').iloc[:,1:]

    return inputs_training,outputs_training


inputs_train, outputs_train = createDataSet(8)

delay = 8
# inputs_train, outputs_train = readDataSet(delay)

dataArchitechtures = pd.DataFrame(columns=['Architechture', 'Mean_sq', 'Loss'])

# ------------------------------------- Split dataset ----------------------------------------------

# inputs_train, inputs_test,outputs_train,outputs_test = train_test_split(inputs, outputs, test_size=0.1)

ins = inputs_train.shape[1]
outs = outputs_train.shape[1]

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

# ------------------------------------- Model Definition ----------------------------------------------
inputs = tf.keras.Input(shape=(ins,))
x = tf.keras.layers.Dense(64, activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(32, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(16, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(8, activation=tf.nn.relu)(x)


# x = tf.keras.layers.Dense(int(8), activation=tf.nn.relu)(inputs)
# x = tf.keras.layers.Dense(int(2), activation=tf.nn.relu)(x)

#4,3,2
# 8,6,4
#6,4,3
#5,3,2

outputs = tf.keras.layers.Dense(outs)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
 

opt = tf.keras.optimizers.Adam(learning_rate=0.0013)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

# ------------------------------------- Model training ----------------------------------------------
history = model.fit(inputs_train, outputs_train, validation_split=0.3, epochs=300, batch_size=512)
# Graph 
modelPath = "64-32-16-8 130"  

plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['val_loss'], label="val_los s")
plt.title('Model loss - '+str(delay)+" delays, "+modelPath)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.yscale('log')
plt.legend()
plt.tight_layout() # Adjust the padding between and around subplots.
plt.show()    

mkdir(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath)

plt.savefig(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\"+str(delay)+" delays, "+modelPath+".pdf")
plt.savefig(".\\models\\"+str(delay)+" delays\\Models Loss - Graphs\\"+str(delay)+" delays, "+modelPath+".jpg")

plt.close()


#--------------------------------------  Save model -----------------------------------------------

loss = history.history['loss']
val_loss = history.history['val_loss']


model.save(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\ANN_"+modelPath+"_"+str(delay)+"delays.h5")

pd.DataFrame(loss).to_csv(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\loss.csv", sep=';')
pd.DataFrame(val_loss).to_csv(".\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath+"\\val_loss.csv", sep=';')



# # ------------------------------------- Model Evaluation ----------------------------------------------
# loss, mean_sq = model.evaluate(inputs_test,outputs_test)

# dataArchitechtures = dataArchitechtures.append({'Architechture': modelPath, 'Mean_sq': mean_sq, 'Loss':loss}, ignore_index=True)
# SaveData(dataArchitechtures,delay)