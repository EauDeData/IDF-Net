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

from tqdm import tqdm

from src.dataloaders.base import IDFNetDataLoader

def read_img(path):
    img = pdf2image.convert_from_path(path.strip())
    return [np.array(img[i]) for i in range(len(img))]

class BOEDataset(IDFNetDataLoader):

    name = 'boe_dataset'
    def __init__(self, jsons_data_folder, min_height = 224, min_width = 224, device = 'cuda',) -> None:
        super(BOEDataset, self).__init__()
        self.data = []
        self.text = []
        for root, _, files in os.walk(jsons_data_folder):
            for file in tqdm(files, desc=f"Processing dataset..."):
                if not os.path.splitext(file)[1].lower() in ['.json']: continue

                fname = os.path.join(root, file)
                datapoint = json.load(open(fname, 'r'))
                self.data.append(datapoint)

                path = datapoint['path']
                path = os.path.splitext(path)[0]+'.html'
                path = path.replace('images', 'htmls')

                self.text.append(open(path, 'r').read())
                break
        

        self.device = device
        
        self.min_width = min_width
        self.min_height = min_height

        self.tokenizer = 0
    
    def iter_text(self):
        for datapoint in self.text: yield datapoint
    
    def __getitem__(self, idx):
        
        image_json = self.data[idx]
        images = read_img(image_json['path'])
        crops = []

        for page in image_json['pages']:
            num_page = int(page)
            for item in image_json['pages'][page]:
                x1, y1, x2, y2 = item['bbox']
                if (x2 - x1) < self.min_width or (y2 - y1) < self.min_width: continue
                array = images[num_page][y1:y2, x1:x2] / 255
                crops.append(torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).float())
        if not isinstance(self.tokenizer, int): textual = self.tokenizer.predict(self.text[idx])
        else: textual = self.text[idx]
        return {"crops": crops}, textual
        


    

