import numpy as np
import pandas as pd
import cv2
import os
import glob
import xml.etree.ElementTree as ET
from shutil import copyfile

class PreProcessing():
    def __init__(self, path_anno, path_img, path_dest):
        self.path_anno = path_anno
        self.path_img = path_img
        self.path_dest = path_dest
        self.process()
        self.to_dataframe()
        
    dataset = {
            "class" : [],
            "xmin" : [],
            "ymin" : [],
            "xmax" : [],
            "ymax" : [],
            "file_name" : [],
            "defect_x_center" : [],
            "defect_y_center" : [],
            "defect_width" : [],
            "defect_height" : [],
            "image_width" : [],
            "image_height" : [],
            "category_codes" : [],
            "path" : []
        }
    
    def process(self):
        all_files = []
        
        for path, subdirs, files in os.walk(self.path_anno):
            for name_file in files:
                all_files.append(os.path.join(path, name_file))
        #lấy dữ liệu từ các file xml
        for anno in all_files:
            tree = ET.parse(anno)
            cnt = 0
            
            for element in tree.iter():
                file_name_anno = os.path.basename(anno)
                filename = file_name_anno[:-4]
                if 'name' in element.tag:
                    name = element.text
                if 'size' in element.tag:

                    for attr in list(element):
                        if 'width' in attr.tag:
                            image_width = int(round(float(attr.text)))
                        if 'height' in attr.tag:
                            image_height = int(round(float(attr.text)))

                if 'object' in element.tag:
                    cnt += 1
                    for attr in list(element):
                        if 'name' in attr.tag:
                            name = attr.text

                        if 'bndbox' in attr.tag:
                            xmin = -1
                            xmax = -1
                            ymin = -1
                            ymax = -1

                            for position in list(attr):
                                if 'xmin' in position.tag:
                                    xmin = int(round(float(position.text)))
                                if 'ymin' in position.tag:
                                    ymin = int(round(float(position.text)))
                                if 'xmax' in position.tag:
                                    xmax = int(round(float(position.text)))
                                if 'ymax' in position.tag:
                                    ymax = int(round(float(position.text)))

                    self.dataset['class'] += [name]
                    if name == 'missing_hole':
                        self.dataset['category_codes'] += [0]
                    elif name == 'mouse_bite':
                        self.dataset['category_codes'] += [1]
                    elif name == 'open_circuit':
                        self.dataset['category_codes'] += [2]
                    elif name == 'short':
                        self.dataset['category_codes'] += [3]
                    elif name == 'spur':
                        self.dataset['category_codes'] += [4]
                    elif name == 'spurious_copper':
                        self.dataset['category_codes'] += [5]

                    self.dataset['image_width'] += [image_width]
                    self.dataset['image_height'] += [image_height]
                    self.dataset['file_name'] += [filename]
                    self.dataset['xmin'] += [xmin]
                    self.dataset['ymin'] += [ymin]
                    self.dataset['xmax'] += [xmax]
                    self.dataset['ymax'] += [ymax]
                    self.dataset['defect_x_center'] += [((xmin + xmax) / 2) / image_width]
                    self.dataset['defect_y_center'] += [((ymin + ymax) / 2) / image_width]
                    self.dataset['defect_width'] += [(xmax - xmin) / image_width]
                    self.dataset['defect_height'] += [(ymax - ymin) / image_height]  
            for index in range(cnt):
                self.dataset['path'] += [os.path.join(self.path_img, filename + '.jpg')]
    
    def to_dataframe(self):
        data_frame = pd.DataFrame(self.dataset)
        data_frame.to_csv(os.path.join(self.path_dest, 'dataframe.csv'))
        dataframe = pd.read_csv(os.path.join(self.path_dest, 'dataframe.csv'))
        return dataframe


if __name__ == "__main__":
    try:
        os.makedirs('./dest', exist_ok=True)
        path_dest = os.path.abspath('./dest')
    except:
        print('directory probably already existing')
    path_anno = "T:/university course/project/PCD defect detection/dataset/VOC_PCB/Annotations"
    path_img = "T:/university course/project/PCD defect detection/dataset/VOC_PCB/JPEGImages"
    dataset = PreProcessing(path_anno, path_img, path_dest)
    print(dataset.to_dataframe())



