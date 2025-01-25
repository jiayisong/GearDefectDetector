import os
import json
import math
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm



input_dir = 'E:/Dataset/chilun/val.json'
output_dir = 'E:/Dataset/chilun/val_resize.json'

with open(input_dir, 'r') as f1:
    results = json.load(f1)
    for i in tqdm(results['images']):
        name = i['file_name']
        im = cv2.imread('E:/Dataset/chilun/train/' + name)
        h, w, _ = im.shape
        if h == 1500:
            if w == 1000:
                target_size = [1902, 1076]
            elif w == 1400:
                target_size = [1500, 841]
            else:
                raise RuntimeError('img shape error')
        elif h == 2000:
            target_size = [3354, 1262]
        else:
            raise RuntimeError('img shape error')
        im = cv2.resize(im, target_size[::-1], interpolation=cv2.INTER_CUBIC)
        #im[:,:,1] = 0
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blocksize = 1011
        C = 0
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, C)
        r1 = cv2.Canny(gray, 200, 250)
        #im[:, :, 1] = r1
        r2 = cv2.Canny(gray, 100, 150)
        cv2.imshow('origin', im)
        cv2.imshow('binary', r2)
        cv2.imshow('canny', r1)
        cv2.waitKey()



