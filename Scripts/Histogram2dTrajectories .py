import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl

import numpy as np
from numpy.random import multivariate_normal

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


def SaveData(targetsX,targetsY,targetsZ,x,y,z,drone,type): 
    #print('x:'+str(len(x))+'y:'+str(len(y))+'z:'+str(len(z)))

    filename = '.\\DataBase\\Training Trajectories\\TrainingTrajectories-ANN'
    
    sheets = {'target x':targetsX,'target y':targetsY,'target z':targetsZ,'x':x,'y':y,'z':z}
    data = pd.DataFrame(sheets)
    
    with pd.ExcelWriter(filename+'.xlsx',engine='openpyxl', mode='a', if_sheet_exists ='replace') as writer:
            data.to_excel(writer, sheet_name="Training Data - Drone "+str(drone), index=False)

def createDataSet(type): 
    drone = 0
    targetsX =[]
    targetsY =[]
    targetsZ =[]
    x=[]
    y =[]
    z =[]
    
    if type == "Test":     
        limit = 11
    elif type == "Training":
        limit = 31

    
    for trajectory in range (1,limit):
        results, sheets = readAllSheets(".\\DataBase\\"+type+" Trajectories\\"+type+"Trajectory_"+str(trajectory)+"_Results.xlsx")
        # for drone in range (4):

        targetsX.extend(results[sheets[drone]].loc[:,"target x"])
        targetsY.extend(results[sheets[drone]].loc[:,"target y"])
        targetsZ.extend(results[sheets[drone]].loc[:,"target z"])
        x.extend(results[sheets[drone]].loc[:,"x"])
        y.extend(results[sheets[drone]].loc[:,"y"])
        z.extend(results[sheets[drone]].loc[:,"z"])
            
    SaveData(targetsX,targetsY,targetsZ,x,y,z,drone,type)

type = "Training"
createDataSet()
drone = 0
results, sheets = readAllSheets('.\\DataBase\\'+type+' Trajectories\\'+type+'Trajectories-ANN.xlsx')
targetsX = results[sheets[drone]].loc[:,"target x"]
targetsY = results[sheets[drone]].loc[:,"target y"]
targetsZ = results[sheets[drone]].loc[:,"target z"]
x = results[sheets[drone]].loc[:,"x"]
y = results[sheets[drone]].loc[:,"y"]
z = results[sheets[drone]].loc[:,"z"]

fig,ax = plt.subplots(2,3)
values = [[targetsX,targetsY],[targetsX,targetsZ],[targetsY,targetsZ],[x,y],[x,z],[y,z]]
axes = [["Target x(cm)","Target y(cm)"],["Target x(cm)","Target z(cm)"],["Target y(cm)","Target z(cm)"],["x(cm)","y(cm)"],["x(cm)","z(cm)"],["y(cm)","z(cm)"]]
row = 0
column = 0

plt.rcParams.update({  #Style settings
"text.usetex": True,
"font.family": "Palatino",
"axes.titlesize" : 17,
'figure.figsize' : (18.5,9.5),
"font.size": 13
})

fig.suptitle('2D histogram training trajectories - Drone '+str(drone))

for image in range (len(values)):
    if image == 3: 

        row = 1
        column = 0 
        
    rang = [[values[image][0].min(),values[image][0].max()],[values[image][1].min(),values[image][1].max()]]
    h = ax[row,column].hist2d(values[image][0],values[image][1],bins=250,norm = mpl.colors.LogNorm(),range=rang)
    ax[row,column].set_xlabel(axes[image][0])
    ax[row,column].set_ylabel(axes[image][1])
    fig.colorbar(h[3], ax=ax[row, column])

    column = column+1
    
    

# fig.tight_layout()

plt.show()