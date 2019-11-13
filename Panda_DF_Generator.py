# import pandas to use pandas DataFrame 
import pandas as pd 
  
# Write each tuple per line into txt file.
def assignListToTxt(data, filename='train_result.txt'):
    file = open(filename,'w') # Creates file if it doesn't exist.
    for x in data:
        # print(str(data))
        file.write(str(x) + "\n")
    return file

# Convert text file to list of data tuples.
# TODO: Maybe check if data contains yolo version and create 
# list column based on entry.
def convertTxtToList(filename, debug=False):
    data = []
    file = open(filename,"r") 
    yolo_read = file.readlines()
    for line in (yolo_read):
        if (debug):
            print(line)
        data.append(line)
    return data

""" 
Read list of txt files, ["file1.txt", "file2.txt", "file3.txt"]
Write all the data to 1 text file, then convert the data to a list
containing all data entries.
"""
def combineTxtFiles(files):
    full_data = open("combined_data.txt", "w")
    for file in files:
        full_data.write(file.read())
    dataset = convertTxtToList(full_data)
    # TODO: Change txt file tuple to contain entries for each respective yolo.
    # generatePandaDF(dataset)

  
"""
Expecting txt file containing newline separated tuples of pair:
(imageID, average IoU score)
"""
def generatePandaDF(data, debug=False):
    # create DataFrame using data 
    df = pd.DataFrame(data, columns =['ImageID', 'IoU_yoloV3', 'IoU_yoloV2', 'IoU_yolo_tiny']) 
    if (debug):
        print(df)  
    return df