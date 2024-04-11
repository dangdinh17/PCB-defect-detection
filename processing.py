import numpy as np
import pandas as pd
import cv2
import os

from preprocessing import PreProcessing
from shutil import copyfile
from sklearn.model_selection import train_test_split

class DataProcess():
    def __init__(self, data, path_dest):
        self.data = data
        self.save_working = path_dest
        
        self.image_copy_all()
        self.make_labels()
        self.make_yaml()
        self.check_yaml()
        self.split_file()
        self.make_image_path_txt()
        
    #tạo danh sách các label    
    def classes(self):
        classes = sorted(self.data['class'].unique())
        return classes
    
    #sao chép ảnh sang địa chỉ làm việc mới
    def image_copy_all(self):
        self.make_dir()
        try:
            os.makedirs(self.save_working + '/images')
        except:
            pass

        for index in range(len(self.data)):
            file_name  = self.data.loc[index, "file_name"] + '.jpg'
            copyfile(self.data.loc[index,'path'], os.path.join(self.save_working, 'images', file_name))
            self.data.loc[index,'path'] = os.path.join(self.save_working, 'images', file_name)

    # tách dữ liệu train và test
    def split_file(self):
        train_file, val_file = train_test_split(self.data['file_name'].unique(), test_size=0.2, random_state = 42)
        self.train_data = self.data[self.data['file_name'].isin(train_file)]
        self.val_data = self.data[self.data['file_name'].isin(val_file)]
        self.train_data.sort_values('file_name')
        self.val_data.sort_values('file_name')

    # tạo dữ liệu tên file txt cho label
    def make_image_path_txt(self):
        with open(self.save_working + '/train.txt', 'a') as file_pos:
            for path in self.train_data['path'].unique():
                print(path, file = file_pos)
        with open(self.save_working + '/val.txt', 'a') as file_pos:
            for path in self.test_data['path'].unique():
                print(path, file = file_pos)

    #tạo địa chỉ cho các file txt
    def make_dir(self):
        try:
            os.makedirs(self.save_working, exist_ok = True)
        except:
            print('the directory was already made')

    #tạo dữ liệu label
    def make_labels(self):
        self.make_dir()
        try:
            os.removedirs(self.save_working + '/labels')
            print('remove success')
        except:
            print('nothing to remove')

        try:
            os.makedirs(self.save_working + '/labels')
        except:
            print('creating fail')
        #ghi dữ liệu cho label
        for index in range(len(self.data)):
            file_name = self.data.loc[index, "file_name"] 
            try:
                with open(os.path.join(self.save_working, 'labels', file_name + '.txt'), 'a') as file_pos:
                    print(self.data.loc[index,'category_codes'], self.data.loc[index,'defect_x_center'],
                          self.data.loc[index, 'defect_y_center'], self.data.loc[index, 'defect_width'], 
                          self.data.loc[index, 'defect_height'], file = file_pos)
            except:
                print('labels creating fail')

    #tạo file yaml cho model
    def make_yaml(self):
        yaml_file = os.path.join(self.save_working, 'data.yaml')
        with open(yaml_file, 'w') as file_pos:
            file_pos.write('names:\n')
            for index, value in enumerate(self.classes()):
                file_pos.write(f'  {index}: {value}\n')
            file_pos.write(f'nc: {len(self.classes())}\n')
            file_pos.write(f'train: {self.save_working}/train.txt\n')
            file_pos.write(f'val: {self.save_working}/val.txt\n')
        
    #kiểm tra địa chỉ của file yaml
    def check_yaml(self):
        with open(self.save_working + '/data.yaml', 'r') as file_pos:
            print(file_pos.readlines())
    
    #trả về địa chỉ file yaml
    def yaml_file(self):
        yaml_file = os.path.join(self.save_working, 'data.yaml')
        return yaml_file
    
    #khởi tạo tất cả từ đầu
    def reset_all(self):
        os.remove(self.save_working)
