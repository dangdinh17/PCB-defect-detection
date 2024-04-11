import ultralytics
import sys
import cv2
import pandas as pd
import random
import os

from ultralytics import YOLO
from PyQt6 import QtCore, QtGui, QtWidgets, uic
from PyQt6.QtWidgets import *
from PyQt6.uic import loadUi


#Cửa sổ chính
class Main_w(QMainWindow):
    def __init__(self):
        super(Main_w, self).__init__()
        uic.loadUi('ui/mainwindow.ui', self)

        self.auto_bt.clicked.connect(self.auto_mode)
        self.manual_bt.clicked.connect(self.manual_mode)
        
    def auto_mode(self):
        widget.setCurrentIndex(1)
    
    def manual_mode(self):
        widget.setCurrentIndex(2)
        
#Cửa sổ automatic
class Auto_w(QMainWindow):
    def __init__(self):
        super(Auto_w, self).__init__()
        uic.loadUi('ui/automatic.ui', self)
        
        self.back_bt1.clicked.connect(self.back1)
        self.model = YOLOv8("auto", None)
        self.real_path = " "
        self.pred_path = " "
        
        self.pred_bt1.clicked.connect(self.get_path)
        self.pred_bt1.clicked.connect(self.real_img)
        self.pred_bt1.clicked.connect(self.auto_pred)
        self.pred_bt1.clicked.connect(self.pred_display1)
        self.pred_bt1.clicked.connect(self.del_img_path)
        
    def pred_display1(self):
        labels = self.model.get_label()
        for label in labels:
            self.pred_label1.setText(label)
                
    def back1(self):
        widget.setCurrentIndex(0)
    
    def get_path(self):
        self.real_path = self.model.auto_mode()
        self.pred_path = self.model.predict()
        print(self.real_path,self.pred_path)
        
    def real_img(self):   
        input_img = QtGui.QPixmap(self.real_path)
        self.input_img1.setPixmap(input_img)
    
    def auto_pred(self):
        output_img = QtGui.QPixmap(self.pred_path)
        self.output_img1.setPixmap(output_img)
        
    def del_img_path(self):
        if os.path.exists(self.pred_path):
            os.remove(self.pred_path)
        if os.path.exists(self.real_path):
            os.remove(self.real_path)
#Cửa sổ manual
class Manual_w(QMainWindow):
    def __init__(self):
        super(Manual_w, self).__init__()
        uic.loadUi('ui/manual.ui', self)
        
        self.real_path = ""  # Thêm khởi tạo cho real_path
        self.pred_path = ""  # Thêm khởi tạo cho pred_path
        
        self.back_bt2.clicked.connect(self.back2)
        self.open_img.clicked.connect(self.output_img2.clear)
        self.open_img.clicked.connect(self.pred_label2.clear)
        self.open_img.clicked.connect(self.real_img)
        self.pred_bt2.clicked.connect(self.manual_pred)
        self.pred_bt2.clicked.connect(self.del_img_path)
        
    def back2(self):
        widget.setCurrentIndex(0)
        
    def pred_display1(self):
        labels = self.model.get_label()
        for label in labels:
            self.pred_label1.setText(label)
            
    def real_img(self):
        self.real_path = QFileDialog.getOpenFileName(None, 'Mở File',' ','(*.jpg);;(*.png)')
        input_img = QtGui.QPixmap(self.real_path[0])
        self.input_img2.setPixmap(input_img) 

    def manual_pred(self):
        if os.path.exists(self.real_path[0]):
            model = YOLOv8("manual", self.real_path[0])
            self.pred_path = model.predict()
            output_img = QtGui.QPixmap(self.pred_path)
            
            self.output_img2.setPixmap(output_img)
            labels = model.get_label()
            for label in labels:
                self.pred_label2.setText(label)
        
    def del_img_path(self):
        if os.path.exists(self.pred_path):
            os.remove(self.pred_path)
        
class YOLOv8():
    def __init__(self, mode, img_test):
        self.mode = mode
        self.img_test = img_test
        self.result = ''
        self.model = YOLO('weight/best_weight.pt')
    def predict(self):
        self.result = self.model.predict(source = self.img_test, imgsz = 608, conf = 0.6)
        result_path = os.path.abspath(self.result[0].save())
        return result_path
    
    def auto_mode(self):
        self.data_frame = pd.read_csv('dest/dataframe.csv')
        index = random.randint(0, len(self.data_frame))
        self.img_test = self.data_frame.loc[index, 'path']
        img = cv2.imread(self.img_test)
        matching_path = self.data_frame.loc[self.data_frame['path'] == self.img_test]
        
        for index, row in matching_path.iterrows():
            xmin = self.data_frame.loc[index, 'xmin']
            xmax = self.data_frame.loc[index, 'xmax']
            ymin = self.data_frame.loc[index, 'ymin']
            ymax = self.data_frame.loc[index, 'ymax']

            category_defect = self.data_frame.loc[index, 'class']
            # print(xmin, ymin, xmax, ymax, category_defect)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
            cv2.putText(img, category_defect, (xmin, ymin),cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255,255,255), 2, cv2.LINE_AA)
        save_path = os.path.basename(self.img_test)
        cv2.imwrite(save_path, img)    
        return os.path.abspath(save_path)
    
    def get_label(self):
        label = []
        for result in self.result: 
            boxes = result.boxes 
            for box in boxes: 
                box=box.numpy()
                name = self.model.names.get(box.cls.item())
                label.append(name)
        return list(set(label))
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    
    main_w = Main_w()
    auto_w = Auto_w()
    manual_w = Manual_w()
    
    widget.addWidget(main_w)
    widget.addWidget(auto_w)
    widget.addWidget(manual_w)
    
    widget.setFixedSize(1000, 800)
    
    widget.setCurrentIndex(0)
    widget.show()
    app.exec()