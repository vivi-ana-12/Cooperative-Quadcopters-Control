import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .ExcelFileManager import ExcelFileManager

class GraphVisualizer:
    def __init__(self,quadcopters_number):
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Palatino",
            "axes.titlesize": 17,
            'figure.figsize': (18.5, 9.5),
            "font.size": 13
        })
    
    def plotSimpleDataset(self, trajectoryNumber,load,saveGraph,trajectoryType):
        variables = ["x(m)", "y(m)", "z(m)", "Error x", "Error y", "Error z"]
        if trajectoryType:     
            filename = ".\\DataBase\\Test trajectories\\TestTrajectory_"+str(trajectoryNumber)+"_Results.xlsx"    
        else: 
            filename = ".\\DataBase\\Training Trajectories\\TrainingTrajectory_"+str(trajectoryNumber)+"_Results.xlsx"    
        
        results, sheets = ExcelFileManager.readExcelFile(filename)
        
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


    def plotCompleteDataset(self, trajectoryNumber,saveGraph,trajectoryType):
        delay = 16
        quadcopters_number = 4
        filename_test2 = ".\\DataBase\\Test Results\\Test 1\\Test1_Trajectory"+str(trajectoryNumber)+"_Results.xlsx"    
        filename_Test = ".\\DataBase\\Test Results\\Test_Trajectory"+str(trajectoryNumber)+"_Results.xlsx"

        results, sheets = ExcelFileManager.readExcelFile(filename_Test) # Open and read the file
        results2, sheets2 = ExcelFileManager.readExcelFile(filename_test2) # Open and read the file

        dataset = [[[] for x in range(16)] for y in range(len(sheets))]
        dataset_test2 = [[[] for x in range(16)] for y in range(len(sheets2))] 

        for sheet in range (len(sheets)): # Fill the dataset with the data from the file
            for variable in range (16): 
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
                    ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset[drone][variable+7][delay+2:],"b",linestyle = '--')
                    ax[variable, drone].plot(dataset[drone][0],dataset[drone][variable+1],"r")
            
                else: # In the next 3 rows print the error of each axis
                    ax[variable, drone].plot(dataset[drone][0][delay+2:],dataset[drone][variable+7][delay+2:],"r")
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

        plt.savefig(".\\Graphs\\Test Results\\Trajectory "+str(trajectoryNumber)+"_W.pdf")

        plt.close('all')  #Close all open figures

