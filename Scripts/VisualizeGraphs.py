import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os.path


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

def plotData(trajectoryNumber,load,saveGraph,trajectoryType):
    
    if trajectoryType:     
        filename = ".\\DataBase\\Test trajectories\\TestTrajectory_"+str(trajectoryNumber)+"_Results.xlsx"    
    else: 
        filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectory_"+str(trajectoryNumber)+"_Results.xlsx"    
        
    results, sheets = readAllSheets(filename)
    
    dataset = [[[] for x in range(13)] for y in range(len(sheets))] 
    
    
    for sheet in range (len(sheets)):
        for variable in range (13): 
            dataset[sheet][variable].extend(results[sheets[sheet]].to_numpy().transpose().tolist()[variable])
            
            
    image = 1 #Initialize the image to print
    variables = ["x(m)","y(m)","z(m)","Error x","Error y","Error z"] #Variable names per row
    quadcopters_number = 4
    
    plt.rcParams.update({  #Style settings
    "text.usetex": True,
    "font.family": "Palatino",
    "axes.titlesize" : 17,
    'figure.figsize' : (18.5,9.5),
    "font.size": 13
    })
    
    plt.close('all')  #Close all open figures
    
    fig, ax = plt.subplots(len(variables),quadcopters_number,sharex = True)
    
    for variable in range (6): #Rows 
        for drone in range (quadcopters_number): #Columns
            if(variable < 3): # In the first 3 rows (x,y and z) print the target at the same time
                if (load):
                    ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+4],"g")
                    ax[variable, drone].plot(dataset[drone][0],dataset[drone+4][variable+1],"r")
                    ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+1],"b",linestyle = '--')
                    
                else: 
                    ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+1],"r",dataset[drone][0],dataset[drone][variable+4],"g")
            else:
                   
                if (load):
                        ax[variable, drone].plot(dataset[drone][0],dataset[drone+4][variable+4],"r")
                        ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+4],"b",linestyle = '--')
                        
                else:
                    ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+4],"r")
            
            if (image < 5 ): # Print the drone number in each column
                ax[variable, drone].set_title(r'Quadcopter '+str(image-1))
            
            if (image == 4):
                if (load):
                    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2),Line2D([0], [0], color='blue', lw=2,linestyle = ':')]
                    ax[variable, drone].legend(custom_lines, ['With load', 'Target','Without load'], bbox_to_anchor = (1.05,1.08))   
                else: 
                    custom_lines = [Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='green', lw=2)]
                    ax[variable, drone].legend(custom_lines, ['Simulation', 'Target'], bbox_to_anchor = (1.05,1.08))   
                    
            if (drone == 0): # Print the variable in each row
                ax[variable, drone].set_ylabel(variables[variable])
                
            if (variable == len(variables)-1): 
                ax[variable, drone].set_xlabel(r'Time (s)')  # Add the x-axis label of the last row
    
            ax[variable, drone].grid(linestyle = '--', linewidth = 0.5) # Adds grid
            image = image + 1  # Increase to the next position
        
        
    plt.tight_layout() # Adjust the padding between and around subplots.
    plt.show() # Display all open figures.
    

    if saveGraph:
        if load: 
            suffix = "W"
        else: 
            suffix = "WO"
            
        if trajectoryType:  
            plt.savefig(".\\Graphs\\Test trajectories\\TestTrajectory"+str(trajectoryNumber)+"_"+str(suffix)+".pdf")
        else: 
            plt.savefig(".\\Graphs\\Training trajectories\\TrainingTrajectory"+str(trajectoryNumber)+"_"+str(suffix)+".pdf")

    
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# trajectoryType = False # Training False, Test True
# trajectoryNumber = 31
# saveGraph = True
# load = False
# plotData(trajectoryNumber,load,saveGraph,trajectoryType)


