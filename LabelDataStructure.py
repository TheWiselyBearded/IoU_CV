from __future__ import print_function

class LabelDataStructure:

    def parseTextFileToDataStructure(self, filename):
        labels_file  = open(filename, "r")
        label_read = labels_file.readlines()
        labels = []
        for line in label_read:
            data = line.split(': ', 1)
            labels.append(data[1])
        print(*labels,sep='\n')
        return labels


# if __name__ == '__main__':
#     lds = LabelDataStructure()
#     lds.parseTextFileToDataStructure("data/coco_labels.txt")
#     lds.parseTextFileToDataStructure("data/Darknet_COCO_labels.txt")    
