from __future__ import print_function
import json
import LabelDataStructure


class IoU_Calculator:
    '''
    COCO dataset is split into two parts. The annotations and the images section. The annotations section contains a bounding
    box of the objects in absolute pixel coordinates. The images section contains the image width and height as pixel values. 

    In the following blocks of code, I create the annotations dictionary and image dictionary in order to normalize the 
    annotation bounding boxes from absolute pixel coordinates to relative pixel coordinates (x=100, y= 100) -> (x=.5,y=.5) 
    Relative pixel coordinates means the relative position of the point regards to the image. 
    If x=.5 and y=.5 it is 50% of the width, 50% of the height at the midpoint of the image.

    The reason is that we cannot ensure that the image will always be the same size as the annotations. It might be
    corrupted or slightly smaller. In those situations the bounding boxes will be completely wrong.
    '''
    def __init__(self):
        # Turn to dictionary, because look up time is 1
        self.annotations_dict = {}
        self.image_dict = {}
        self.result_dict = {}
        self.data = ""


        
    def load_JSON(self, filename="data/json/instances_train2014.json"):
        with open(filename, "r") as read_file:
            self.data = json.load(read_file)


    '''
    The following code loads the results.txt from our darknet running. I add these values into a dictionary 
    and parse them accordingly to match the json information. This code goes through the results.txt line by line.
    If the image name is in the annotations dictionary, it sets the flag to true. The following lines are added as 
    values to the image name key. 

    If the flag is false, then following lines don't get added.
    '''
    def create_result_dataset(self, filename="data/result.txt"):
        result_yolov3 = open(filename, "r")
        result_yolov3_read = result_yolov3.readlines()

        count_of_images = 0
        num_in_annot = 0
        found = False
        id_found = ""
        for line in (result_yolov3_read):
            if "image_name:" in line:
                split_line = line.split(" ")
                split_line_id = int(split_line[1].lstrip('0'))
                #print(split_line_id)                
                if (split_line_id in self.annotations_dict):
                    num_in_annot = num_in_annot + 1                    
                    id_found = split_line_id
                    found = True
                else:
                    found = False
                count_of_images = count_of_images + 1
            else:
                if (found): # Parsing                    
                    line_class_bbox = line.split(",")
                    classvalue = line_class_bbox[0].split(" ")[1]
                    bboxes = line_class_bbox[1].split(" ")
                    x = bboxes[1].split(":")[1]
                    y = bboxes[2].split(":")[1]
                    w = bboxes[3].split(":")[1]
                    h = bboxes[4].split(":")[1][0:-2] #Get rid of the \n
                    parsed_line = [classvalue, x, y, w, h]
                    if (self.result_dict.get(id_found) == None):
                        self.result_dict[int(id_found)] = [parsed_line]
                    else:
                        self.result_dict[id_found].append(parsed_line)          
    
    
    '''
    Creates the image dataset with information of the images. I have commented the 2014 and 2017 jsons. 
    These json files follow different formats. Choose either depending on which version of the instance.json you are using.

    image_dict(Key = ImageID) = {Values, FileName, Image Height, Image Width, CocoURL}

    CocoURL is used for downloading the image form the internet to use in ImageVerify.
    '''
    def create_image_dataset(self):
        for i in range(len(self.data['images'])):
            # For 2014 Json
            file_name = self.data['images'][i]['file_name'][15:]            
            # For 2017 Json
            #file_name = data['images'][i]['file_name']
            image_id_name = int(''.join([ i.lstrip('0') for i in file_name ]).split('.')[0])
            im_height = self.data['images'][i]['height']
            im_width = self.data['images'][i]['width']
            url = self.data['images'][i]['coco_url']   
            self.image_dict[image_id_name] = {"filename": file_name, "height":im_height, "width":im_width, "coco_url":url}


    ''' 
    Creates the annotation dataset with information of the images. 

    I check if the image id is inside the image set.
    If it is, I normalize the annotation bounding box with the image data set's width and height of the image.

    I need to check if the image id is in the image set in case there is some mistake in the annotations or missing data.

    Annotations_dict[Key = ImageID] = [Value, An array of dictionaries containing the {Category ID and Bounding Box information}]
    '''
    def create_annotations_dataset(self):
        annotLength = len(self.data['annotations'])
        print("Annotatioan")
        print(self.data['annotations'][4])
        for i in range(annotLength):
            image_id = self.data['annotations'][i]['image_id']
            bbox = self.data['annotations'][i]['bbox']
            category_id = self.data['annotations'][i]['category_id']
            if (image_id in self.image_dict):
                im_width = self.image_dict[image_id]['width']
                im_height = self.image_dict[image_id]['height']
                bbox = [bbox[0]/im_width, bbox[1]/im_height, bbox[2]/im_width, bbox[3]/im_height]
                # print(image_id)                
                if self.annotations_dict.get(image_id) == None:
                    self.annotations_dict[image_id] = [{"category_id":category_id, "bbox": bbox}]
                else:
                    self.annotations_dict[image_id].append({"category_id":category_id, "bbox": bbox})

    def groundTruthParse(self, bbox_gt):
        x_gt=float(bbox_gt[0])
        y_gt=float(bbox_gt[1])
        w_gt=float(bbox_gt[2])
        h_gt=float(bbox_gt[3])
        
        left_gt = float(x_gt)
        right_gt = float((x_gt + w_gt))
        top_gt = float(y_gt)
        bot_gt = float((y_gt + h_gt))
        print(left_gt, top_gt, right_gt, bot_gt)
        box_groundtruth = [left_gt, top_gt, right_gt, bot_gt]
        return box_groundtruth

    def yoloBoxParse(self, bbox):
        x=float(bbox[1])
        y=float(bbox[2])
        w=float(bbox[3])
        h=float(bbox[4])

        left = float((x - w/2))
        right = float((x + w/2))
        top = float((y - h/2))
        bot = float((y + h/2))

        box_yolo = [left, top, right, bot]
        return box_yolo

    '''
    IOU CALCULATION FUNCTION
    https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc

    Takes in two bounding boxes, each with format [left, top, right, bot]
    Returns the IOU score.
    ''' 
    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
    
    def parseTextFileToDataStructure(self, filename):
        labels_file  = open(filename, "r")
        label_read = labels_file.readlines()
        labels = []
        for line in label_read:
            data = line.split(': ', 1)
            labels.append(data[1])
        # print(*labels,sep='\n')
        return labels


    
if __name__ == '__main__':
    iou = IoU_Calculator()
    # Store labels in memory as lists.
    coco_labels = iou.parseTextFileToDataStructure("data/coco_labels.txt")
    darknet_coco_labels = iou.parseTextFileToDataStructure("data/Darknet_COCO_labels.txt") 
    # Setup IoU data structures.
    iou.load_JSON()
    iou.create_image_dataset()
    iou.create_annotations_dataset()   
    iou.create_result_dataset()
    '''
    SINGLE IMAGE IOU CALCULATIONS
    The coordinates need to follow the conversions:
    left_gt,top_gt,right_gt,bot_gt as well as the conversions for left,top,right,bot.

    Ground truth format from the annotations: top left corner x, top left corner y, width, height
    Our Darknet format from the our result_dict: midpoint x, midpoint y, width, height
    '''
    # Ground Truth Values
    bbox_gt = iou.annotations_dict[374458][0]['bbox']
    box_groundtruth = iou.groundTruthParse(bbox_gt)
    box_yolo = iou.yoloBoxParse(iou.result_dict[374458][0])
    iou_score = iou.bb_intersection_over_union(box_groundtruth,box_yolo)
    print(iou_score)