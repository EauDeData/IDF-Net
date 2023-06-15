import os
from typing import *

import numpy as np
import random
import torch
import torchvision
import json
import cv2
import pandas as pd
import string
import re
import pdf2image
from bs4 import BeautifulSoup 
import torch

from tqdm import tqdm

from src.dataloaders.base import IDFNetDataLoader

def read_img(path):
    img = pdf2image.convert_from_path(path.strip())
    return [np.array(img[i]) for i in range(len(img))]

class BOEDatasetOCRd:

    name = 'boe_dataset'
    def __init__(self, jsons_data_folder, min_height = 224, min_width = 224, scale = 0.5,device = 'cuda', replace_path_expression = "('data1tbsdd', 'data2fast/users/amolina')") -> None:
        super(BOEDatasetOCRd, self).__init__()
        self.data = []
        self.text = []
        self.scale = scale


        self.replace = eval(replace_path_expression)            
        
        self.min_width = min_width
        self.min_height = min_height
        self.data = []

        for root, _, files in os.walk(jsons_data_folder):
            for file in tqdm(files, desc=f"Processing dataset..."):
                if not os.path.splitext(file)[1].lower() in ['.json']: continue

                fname = os.path.join(root, file)
                datapoint = json.load(open(fname, 'r'))
                for page in datapoint['pages']:
                    for item in datapoint['pages'][page]:
                        x1, y1, x2, y2 = item['bbox']
                        if ((x2 - x1) > self.min_width) and ((y2 - y1) > self.min_height) and 'ocr' in item:

                            self.data.append({
                                              'file_uuid': datapoint['file'].replace('.pdf', ''), 
                                              'root': datapoint['path'].replace(*self.replace),
                                               'page': page,
                                               'bbx': item['bbox'],
                                               'text': item['ocr']
                                            })
        print(len(self.data))
        self.device = device
        


        self.tokenizer = 0
        self.max_crops = 50
    
    def iter_text(self):
        for datapoint in self.data: yield datapoint['text']
    
    def __len__(self):
        return len(self.text)


    def __getitem__(self, idx):
        pass