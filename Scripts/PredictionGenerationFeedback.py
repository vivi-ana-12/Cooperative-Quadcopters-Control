import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy

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


def SaveData(predictions,trajectory_number,modelPath): 
    #print('x:'+str(len(x))+'y:'+str(len(y))+'z:'+str(len(z)))

    # filename = '..\\Models\\8 delays\\8 delays, '+modelPath+'\\'+modelPath+' - Test Data'
    filename = '..\\Models\\4 delays\\4 delays, '+modelPath+'\\'+modelPath+' - Test Data'
    
    sheets = {'prediction x':predictions.iloc[:,0],'prediction y':predictions.iloc[:,1],'prediction z':predictions.iloc[:,2]}
    data = pd.DataFrame(sheets)
    
    with pd.ExcelWriter(filename+'.xlsx',engine='openpyxl', mode='a', if_sheet_exists ='overlay') as writer:
            data.to_excel(writer, sheet_name=str(trajectory_number), index=False)
            

def ReadData(modelPath,trajectory_number): 
    results, sheets = readAllSheets("..\\DataBase\\Test trajectories\\TestTrajectory_"+str(trajectory_number)+"_Results.xlsx")
    # results[sheets[0]].iloc[:,1:3] = results[sheets[0]].iloc[:,1:3]*5
    # results[sheets[0]].iloc[:,4:6] = results[sheets[0]].iloc[:,4:6]*5

    predictions, sheet= readAllSheets('..\\Models\\4 delays\\4 delays, '+modelPath+'\\'+modelPath+' - Test Data.xlsx')

    delay = 4

    
    trajectory_number =  trajectory_number -1
    predictions = predictions[sheet[trajectory_number]]
    time = results[sheets[0]].iloc[:,0]
    fig, ax = plt.subplots(3)
    ax[0].set_title("x")
    ax[0].plot(time[delay+2:],predictions.iloc[:,0],label='Prediction')
    ax[0].plot(time,results[sheets[0]].iloc[:,1],label='Target')
    ax[0].plot(time,results[sheets[0]].iloc[:,4],label='Input') 
    
    ax[0].legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    
    ax[1].set_title("y")
    ax[1].plot(time[delay+2:],predictions.iloc[:,1])
    ax[1].plot(time,results[sheets[0]].iloc[:,2])
    ax[1].plot(time,results[sheets[0]].iloc[:,5])
    
    
    ax[2].set_title("z")
    ax[2].plot(time[delay+2:],predictions.iloc[:,2])
    ax[2].plot(time,results[sheets[0]].iloc[:,3])
    ax[2].plot(time,results[sheets[0]].iloc[:,6])


    

    
def Predict(modelPath):
    delay = 16
    # model = tf.keras.models.load_model("..\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+modelPath +"\\ANN_"+ modelPath +"_"+str(delay)+"delays.h5")
    model = tf.keras.models.load_model("..\\models\\ANN_64-32-16-8 Fine-Tuning 2_16delays.h5")
    print(model.summary())
    
    # for trajectory_number in range (1,11):
    trajectory_number = 5
    results, sheets = readAllSheets("..\\DataBase\\Test trajectories\\TestTrajectory_"+str(trajectory_number)+"_Results.xlsx")
    results[sheets[0]] = results[sheets[0]].query('index%2 != 0')

    # results[sheets[0]].loc[:,"target x"] =  results[sheets[0]].loc[:,"target x"] - results[sheets[0]].loc[1,"target x"]
    # results[sheets[0]].loc[:,"target y"] =  results[sheets[0]].loc[:,"target y"] - results[sheets[0]].loc[1,"target y"]
    # results[sheets[0]].loc[:,"x"] =  results[sheets[0]].loc[:,"x"] - results[sheets[0]].loc[1,"x"]
    # results[sheets[0]].loc[:,"y"] =  results[sheets[0]].loc[:,"y"] - results[sheets[0]].loc[1,"y"]
    
    # results[sheets[0]].iloc[:,1:3] = results[sheets[0]].iloc[:,1:3]*5
    # results[sheets[0]].iloc[:,4:6] = results[sheets[0]].iloc[:,4:6]*5
    
    
    
    actualState = delay +2
            
    data = results[sheets[0]].values 
    
    
    actual = [[],[],[]]
    previous =  [[],[],[]]
    predictions = pd.DataFrame()
    
    #New 
    actual[0] = data[actualState-delay-1:actualState,4].tolist()[::-1]
    actual[1] = data[actualState-delay-1:actualState,5].tolist()[::-1]
    actual[2] = data[actualState-delay-1:actualState,6].tolist()[::-1]

            
    # previous = actual
    previous[0] = [0]*(delay+1)
    previous[1] = [0]*(delay+1)
    previous[2] = [0]*(delay+1)
    
    for actualState in range (delay+2,len(results[sheets[0]].iloc[:,0])):
        
        actualInputs = actual[0]+actual[1]+actual[2]+previous[0]+previous[1]+previous[2]
    
        predictions = pd.concat([predictions, pd.DataFrame(model.predict([actualInputs],verbose=0))], ignore_index=True,sort = False)

        previous[0] = [predictions.iloc[-1][0]]+ previous[0][:-1]
        previous[1] = [predictions.iloc[-1][1]]+ previous[1][:-1]
        previous[2] = [predictions.iloc[-1][2]]+ previous[2][:-1]
        
        actual[0] = [data[actualState,4]] + actual[0][:-1]
        actual[1] = [data[actualState,5]] + actual[1][:-1]
        actual[2] = [data[actualState,6]] + actual[2][:-1]
        
        for element in range (delay+1): 
            previous[0][element] = float(previous[0][element])
            previous[1][element] = float(previous[1][element])
            previous[2][element] = float(previous[2][element])

            actual[0][element] = float(actual[0][element])
            actual[1][element] = float(actual[1][element])
            actual[2][element] = float(actual[2][element])

    # results[sheets[0]].loc[:,"target x"] =  results[sheets[0]].loc[:,"target x"] + results[sheets[0]].loc[1,"target x"]
    # results[sheets[0]].loc[:,"target y"] =  results[sheets[0]].loc[:,"target y"] + results[sheets[0]].loc[1,"target y"]
    # results[sheets[0]].loc[:,"x"] =  results[sheets[0]].loc[:,"x"] + results[sheets[0]].loc[1,"x"]
    # results[sheets[0]].loc[:,"y"] =  results[sheets[0]].loc[:,"y"] + results[sheets[0]].loc[1,"y"]
    
    # predictions.loc[:,0] = predictions.loc[:,0] + results[sheets[0]].loc[1,"x"]
    # predictions.loc[:,1] = predictions.loc[:,1] + results[sheets[0]].loc[1,"y"]

    time = results[sheets[0]].iloc[:,0]
    fig, ax = plt.subplots(3)
    ax[0].set_title("x")
    ax[0].plot(time[delay+2:],predictions.iloc[:,0],label='Prediction')
    ax[0].plot(time,results[sheets[0]].iloc[:,1],label='Target')
    ax[0].plot(time,results[sheets[0]].iloc[:,4],label='Input') 
    
    ax[0].legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    
    ax[1].set_title("y")
    ax[1].plot(time[delay+2:],predictions.iloc[:,1])
    ax[1].plot(time,results[sheets[0]].iloc[:,2])
    ax[1].plot(time,results[sheets[0]].iloc[:,5])
    
    ax[2].set_title("z")
    ax[2].plot(time[delay+2:],predictions.iloc[:,2])
    ax[2].plot(time,results[sheets[0]].iloc[:,3])
    ax[2].plot(time,results[sheets[0]].iloc[:,6])
    
    # plt.savefig("..\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+ modelPath +"\\"+modelPath+" Test "+str(trajectory_number)+"multiplied.pdf")
    plt.savefig("..\\models\\"+str(delay)+" delays\\"+str(delay)+" delays, "+ modelPath +"\\"+modelPath+" Test "+str(trajectory_number)+".pdf")
    
    # plt.savefig("..\\Models\\"+str(delay)+" delays\\Tests\\PDF\\"+str(delay)+" delays, "+modelPath+" Test "+str(trajectory_number)+".pdf")
    # plt.savefig("..\\Models\\"+str(delay)+" delays\\Tests\\jpg\\"+str(delay      delays, "+modelPath+" Test "+str(trajectory_number)+"multiplied.jpg")
    plt.savefig("..\\Models\\"+str(delay)+" delays\\Tests\\jpg\\"+str(delay)+" delays, "+modelPath+" Test "+str(trajectory_number)+".jpg")

    # plt.close(fig)
    # SaveData(predictions,trajectory_number,modelPath)
        
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# ins = 54
# modelPath = str(int(8*ins))+"-"+str(int(6*ins))

modelPath = "64-32-16-8 130 FineTuning - 1"  
# modelPath = str(int(8*ins))+"-"+str(int(6*ins))+" v6"

Predict(modelPath)
    
# for trajectory_number in range (1,11):
#     ReadData(modelPath,trajectory_number)
