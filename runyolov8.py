import numpy as np
import pandas as pd
import cv2
import json
import os
import glob
import re
import ultralytics
import random

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

from ultralytics import YOLO
from shutil import copyfile
from sklearn.model_selection import train_test_split
from preprocessing import PreProcessing
from processing import DataProcess


class YOLOv8():
    def __init__(self, weight, path_anno, path_img, path_dest, img_test):
        self.weight = weight
        self.path_anno = path_anno
        self.path_img = path_img
        self.path_dest = path_dest
        self.img_test = img_test
        
        # self.load_data()
        self.load_model()
        
    def load_data(self):
        self.dataframe = PreProcessing(self.path_anno, self.path_img, self.path_dest)
        self.dataset = DataProcess(self.dataframe.to_dataframe(), self.path_dest)
        self.yaml_file = self.dataset.yaml_file()

    def load_model(self):
        self.model = YOLO(self.weight)
        
    def train(self):
        self.model.train(data = self.yaml_file,
                    epochs = 5, imgsz = 600, batch = 64,
                    lr0 = 0.0001, dropout = 0.25)
    
    def predict(self):
        data_frame = self.dataframe.to_dataframe()
        index = random.randint(0, len(data_frame))
        img_path = data_frame.loc[index, 'path']
        img = cv2.imread(img_path)
        matching_path = data_frame.loc[data_frame['path'] == img_path]
        print(matching_path)
        for index, row in matching_path.iterrows():
            xmin = data_frame.loc[index, 'xmin']
            xmax = data_frame.loc[index, 'xmax']
            ymin = data_frame.loc[index, 'ymin']
            ymax = data_frame.loc[index, 'ymax']

            category_defect = data_frame.loc[index, 'class']
            print(xmin, ymin, xmax, ymax, category_defect)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(img, category_defect, (xmin, ymin),cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("true defect ", img)
        
        result = self.model.predict(source = img_path, imgsz = 608, conf = 0.6)
        cv2.imshow("predicted defect", result[0].plot(line_width=3))
        
  
if __name__ == "__main__":
    try:
        os.makedirs('./dest', exist_ok=True)
        path_dest = os.path.abspath('./dest')
    except:
        print('directory probably already existing')
    path_anno = "./VOC_PCB/Annotations"
    path_img = "./VOC_PCB/JPEGImages"
    weight = "./weight/best_weight.pt"
    model = YOLOv8(weight, path_anno, path_img, path_dest, None)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()