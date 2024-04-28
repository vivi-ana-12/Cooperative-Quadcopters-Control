import pandas as pd
import os.path


class ExcelFileManager:        
    def readExcelFile(filename):
        if not os.path.isfile(filename):
            return None
        
        xls = pd.ExcelFile(filename)
        sheets = xls.sheet_names
        results = {}
        for sheet in sheets:
            results[sheet] = xls.parse(sheet)
            
        xls.close()
        
        return results, sheets
    
    def exportSimpleDataset(self,dataset, load, trajectoryType, trajectoryNumber):
        if trajectoryType:
            filename = ".\\DataBase\\Test trajectories\\TestTrajectory_" + str(trajectoryNumber) + "_Results.xlsx"
        else:
            filename = ".\\DataBase\\Training trajectories\\TrainingTrajectory_" + str(trajectoryNumber) + "_Results.xlsx"
    
        if not load:
            writer = pd.ExcelWriter(filename, engine='openpyxl')  # Create the document or overwrite it

        sheets = []
    
        for i in range(4):  # Create a sheet for each drone
            sheets.append({'t': dataset[i][9],
                            'x': dataset[i][0], 'y': dataset[i][1], 'z': dataset[i][2],
                            'target x': dataset[i][3], 'target y': dataset[i][4], 'target z': dataset[i][5],
                            'betaE': dataset[i][6], 'alphaE': dataset[i][7], 'zE': dataset[i][8],
                            'thrust': dataset[i][10], 'betaCorr': dataset[i][11], 'rotCorr': dataset[i][12],
                            })
            data = pd.DataFrame(sheets[i])  # Create a DataFrame
            if not load:
                data.to_excel(writer, sheet_name="wo - Drone " + str(i), index=False)  # Write data to the sheet
            else:
                with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                    data.to_excel(writer, sheet_name="w - Drone " + str(i), index=False)  # Create the sheets or overwrite them
    
        if not load:  # Close the writer if it was created
            writer.close()
            
    def exportCompleteDataset(self,dataset, load, trajectoryType, trajectoryNumber, allPredictions):

        sheets = []

        filename = ".\\DataBase\\Test Results\\Test_Trajectory"+str(trajectoryNumber)+"_Results.xlsx"

        writer = pd.ExcelWriter(filename, engine='openpyxl') #Create the document or overwrite it
        
        for i in range (4): # Create a sheet for each drone
            Columns = {'t':dataset[i][9],'x':dataset[i][0],'y':dataset[i][1],'z':dataset[i][2],
                          'target x':dataset[i][3],'target y':dataset[i][4],'target z':dataset[i][5],
                          'prediction x': allPredictions[i][0], 'prediction y': allPredictions[i][1],'prediction z': allPredictions[i][2],
                          'betaE':dataset[i][6],'alphaE':dataset[i][7],'zE':dataset[i][8],
                          'thrust':dataset[i][10],'betaCorr':dataset[i][11],'rotCorr':dataset[i][12]}
            

            sheets.append(Columns)
            
            data = pd.DataFrame(sheets[i])
            
            data.to_excel(writer, sheet_name="w - Drone "+str(i), index=False) 
            
        writer.close()  
