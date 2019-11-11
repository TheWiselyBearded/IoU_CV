from __future__ import print_function

def parseTextFileToDataStructure(filename):
    labels_file  = open(filename, "r")
    label_read = labels_file.readlines()
    labels = []
    for line in label_read:
        data = line.split(': ', 1)
        labels.append(data[1])
    print(*labels,sep='\n')
  

def main():
    parseTextFileToDataStructure("data/coco_labels.txt")
    parseTextFileToDataStructure("data/Darknet_COCO_labels.txt")

if __name__ == '__main__':
    main()
