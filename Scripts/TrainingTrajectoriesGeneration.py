import matplotlib.pyplot as plt
import math
import numpy as np
import random 
import pandas as pd
import os.path



def SaveData(x,y,z,number): 
    #print('x:'+str(len(x))+'y:'+str(len(y))+'z:'+str(len(z)))

    filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories"
    sheets = {'x':x,'y':y,'z':z}
    data = pd.DataFrame(sheets)
    
    with pd.ExcelWriter(filename+".xlsx",engine='openpyxl', mode='a', if_sheet_exists ='replace') as writer:
            data.to_excel(writer, sheet_name=str(number), index=False)

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
    
def plotData(trajectory):
    
    plt.rcParams.update({  #Style settings
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.titlesize" : 17,
    'figure.figsize' : (18.5,9.5),
    "font.size": 13
    })
    
    
    filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectories"
    results, sheets = readAllSheets(filename+".xlsx")
    
    dataset = [[[] for x in range(3)] for y in range(len(sheets))] 
    
    
    for sheet in range (len(sheets)):
        for variable in range (3): 
            dataset[sheet][variable].extend(results[sheets[sheet]].to_numpy().transpose().tolist()[variable])
    
   # plt.close('all')  #Close all open figures
    fig, ax = plt.subplots(3,sharex = True)
    
    values = dataset[trajectory-1]
    tittles = ["x","y","z"]
    
    ax[0].set_title("Trajectory "+str(trajectory))
    for graph in range (3):
        ax[graph].plot(values[graph])
        ax[graph].set_ylabel(tittles[graph]+"(m)")
    ax[2].set_xlabel("t(s)")
    
    
    
# ----------------------------------------------------------------------------------------------------------------
def SteppedRampSign(z):
    signal = []
    value = 0
    time = random.randint(30,50)
    if z :
        step_height = random.uniform(-0.5,3)
    else:
        step_height = random.uniform(-0.5,0.5)
        
    time_step = random.randint(5,math.trunc(time/2))
    
    for t in range (time): 
        signal.append(value)
        if t%time_step == 0:
            value = value+step_height
    return signal 
    
def SteppedTriangularSign(z):
    signal = SteppedRampSign(z)
    down = signal.copy()
    
    for i in range (len(down)): 
        down[i] = -down[i] + signal[len(signal)-1]
    
    signal.extend(down)
    return signal

def StepSignal(z):
    wait_time = random.randint(30,40)
    time_up = random.randint(30,50)
    if z: 
        step_height =  random.uniform(0,4)
    else: 
        step_height =  random.uniform(0,0.5)
        
    value = 0
    signal = []
    
    for t in range(wait_time+1):
        signal.append(value)
        
    value = value + step_height
    for x in range (time_up+1):
        signal.append(value)
        
    value = value - step_height
    
    for x in range (wait_time+1):
        signal.append(value)
        
    return signal 

def PulseWaveSignal(z):
    signal=[]
    if z: 
        time = random.randint(1,100)
        step_height =  random.uniform(1,3)
    else: 
        time = random.randint(1,50)
        step_height =  random.uniform(0,1)
        
    
    time_down= random.randint(30,50)
    time_up= random.randint(30,50)
    
    value = 0

    for t in range (time):
        for t in range(time_down+1):
            signal.append(value)
            
        value = value + step_height
        
        for x in range (time_up+1):
            signal.append(value)
        
        value = value - step_height

            
        if (len(signal)+time_down+time_up+1 >= time):
            while len(signal)<= time:    
                signal.append(value)
            break
    return signal

def PulseWaveRampSignal(z):
    if z: 
        time = random.randint(30,50)
        step_height =  random.uniform(-1,1)
    else: 
        time = random.randint(1,15)
        step_height =  random.uniform(0.1,0.5)
    
    step_width =  random.randint(8,15)
    value = 0
    signal = [value]
    multiplicator = 1
   
    while len(signal) < time: 
        for t in range(step_width):
            signal.append(value)
        value = value + step_height*multiplicator
        
        for x in range (step_width):
            signal.append(value)
        
        value = 0
        
        multiplicator = multiplicator + 0.5

    return signal

def SinWave(z):
    if z: 
        A = random.uniform(0.1,2)
    else:
        A = random.uniform(0.1,0.7)
    f = random.uniform(0.5,1)
    fs = random.uniform(10,15)
    phi = 0
    t = random.uniform(10,5)
    Ts = 1/fs
    n = t / Ts
    n = np.arange(n)
    y = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
    return y


def PulseWaveSineSignal(z):
    signal = []     
    sine = SinWave(z)
    step_width = random.randint(1,math.trunc(len(sine)/2))
    value = sine[0]
       
    for time in range(len(sine)): 
        if (time%step_width == 0):
            value = sine[time]
        signal.append(value) 
    return signal

def RampSignal(z):
    signal = []
    time = random.randint(1,6)
    step = random.uniform(-0.05,0.05)
    value = 0
    
    for t in range(time): 
        signal.append(value) 
        value = value + step
    return signal
    
    
    
    

#-----------------------------------------------------------------------------------------------------------------          
#SaveData(np.zeros(240),np.zeros(240),np.zeros(240),1)
def RandomSignal(z,name):
    n = random.randint(0,7)
    
    if n == 0: 
        signal = SteppedRampSign(z)
        used ="SteppedRampSign"
    # elif n == 1: 
    #     signal = SteppedTriangularSign(z)
    #     used ="SteppedTriangularSign"
    elif n == 1: 
        signal = StepSignal(z)
        used = "StepSignal"
    # elif n == 3: 
    #     signal = PulseWaveSignal(z)
    #     used ="PulseWaveSignal"
    else: 
        signal = PulseWaveRampSignal(z)
        used = "PulseWaveRampSignal"
    # else:
    #     signal = SinWave(z)
    #     used = "SinWave"
    # elif n == 6: 
    #     signal = PulseWaveSineSignal(z)
    #     used = "PulseWaveSineSignal"
    # else: 
    #     signal = RampSignal(z)
    #     used = "RampSignal"
    
    print(name+": "+str(used))
    return signal 

def TrajectoryGeneration(number):

    x = RandomSignal(False,"x")
    y = RandomSignal(False,"y")
    z = RandomSignal(True,"z")
    

    while len(x)<240:
        x.extend(RandomSignal(False,"x"))
    while len(y)<240:
        y.extend(RandomSignal(False,"y"))
    while len(z)<240:
        z.extend(RandomSignal(True,"z"))
    
    
    
    
    if len(x)>240:
        final_x = x[0:240]
    if len(y)>240:
        final_y = y[0:240]
    if len(z)>240:
        final_z = []
        wait = random.uniform(5,15)
        offset = random.uniform(0,5)
        for value in range (240):
            if value <= wait:
                offset = 0
                final_z.append(z[value])
            else:
                final_z.append(z[value] + offset)

    plt.close('all')  #Close all open figures
    fig, ax = plt.subplots(3,sharex = True)
    
    values = [final_x,final_y,final_z]
    tittles = ["x","y","z"]
    
    for graph in range (3):
        ax[graph].plot(values[graph])
        ax[graph].set_title(tittles[graph])
    
    SaveData(final_x,final_y,final_z,number)
    
TrajectoryGeneration(117)
# plotData(28)

# for trajectory in range (1,12): 
#     plotData(12)

# ----------------------------------------------------------------------------
#TrajectoryGeneration(12)
# plotData(1)

# for trajectory in range (1,12): 
#     plotData(12)