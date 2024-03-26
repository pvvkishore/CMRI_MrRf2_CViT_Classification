#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 12:29:12 2023

@author: root
"""

import os
import shutil
folder_path = "/home/user/Desktop/2023MRI_Image_DL/Dataset_Training"
destination_folder = "/home/user/Desktop/2023MRI_Image_DL/Dataset_Training_Mini"
all_files = os.listdir(folder_path)
for subfolder_name in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder_name)
    for subfolder_name in os.listdir(subfolder_path):
        subfolder_path_1 = os.path.join(subfolder_path,subfolder_name)
        all_files_img = os.listdir(subfolder_path_1)
        image_files = [file for file in all_files_img if file.lower().endswith(('.jpg'))]
        num_images = min(50,len(image_files))
        for i in range(num_images):
            image = image_files[i]
            source_path = os.path.join(subfolder_path_1,image)
            for subfolder_name_d in os.listdir(destination_folder):
                subfolder_path_d = os.path.join(destination_folder,subfolder_name_d)
                for subfolder_name_d1 in os.listdir(subfolder_path_d):
                    subfolder_path_d1 = os.path.join(subfolder_path_d,subfolder_name_d1)
                    destin_path = os.path.join(subfolder_path_d1,image)
                    shutil.copy2(source_path,destin_path)
                    print(f"copying:{image}")