from __future__ import print_function
from ast import literal_eval
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
        data.append(literal_eval(line))
    return data

def find_ID_tuple(id, L):
    for idx, val in enumerate(L):
        # print("index is %d and value is %s" % (idx, val))        
        if (str(val[0]) == str(id)):
            return idx
    return -1

""" 
Read list of txt files, ["file1.txt", "file2.txt", "file3.txt"]
Write all the data to 1 text file, then convert the data to a list
containing all data entries.
"""
def combineTxtFiles(files):
    full_data = open("combined_data.txt", "w")
    for file in files:
        full_data.write(file.read())
    return full_data

  
"""
Expecting txt file containing newline separated tuples of pair:
(imageID, average IoU score)
"""
def generatePandaDF(data, debug=False):
    # create DataFrame using data 
    df = pd.DataFrame(data, columns =['ImageID', 'IoU_yoloV3', 'IoU_yoloV2', 'IoU_yolo_tiny']) 
    if (debug):
        print(df.to_string())  
    return df

def combineLists(yolov3, yolov2, yolo_t):
    # Generate list of maximum needed size
    max_size = max(len(yolov3), len(yolov2), len(yolo_t))
    iou_data = [None] * max_size
    # Iterate over lists and update elements based on image ID.
    for x in range(max_size):
        # Assign iou score from v3
        tuple_el = [yolov3[x][0],yolov3[x][1],0,0]
        # Find index for matching image ID from y3 for y2 and tiny.
        idx = find_ID_tuple(yolov3[x][0], yolov2)   
        idx_t = find_ID_tuple(yolov3[x][0], yolo_t)
        if (idx != -1):
            tuple_el[2] = yolov2[idx][1]      # Assign iou score from v2
        if (idx_t != -1):
            tuple_el[3] = yolo_t[idx_t][1]    # Assign iou score from tiny
        # Assign tuple element.
        iou_data[x] = tuple(tuple_el)
    return iou_data



if __name__ == "__main__":
    files = ['yolo_tiny_v3.txt','yolo_v2.txt','yolo_v3.txt']

    y3 = convertTxtToList(files[2]) 
    y2 = convertTxtToList(files[1]) 
    y_tiny = convertTxtToList(files[0]) 
    iou_data = combineLists(y3,y2,y_tiny)    
    # print(*iou_data,sep='\n')
    df = generatePandaDF(iou_data, True)
    df.to_pickle("./iou_scores.pkl")
    df.to_csv("./iou_scores.csv",index=False)





    