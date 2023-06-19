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
    def __init__(self, jsons_data_folder, min_height = 224, min_width = 224, scale = 0.5,device = 'cuda', max_imsize = 64, replace_path_expression = "('data1tbsdd', 'data2fast/users/amolina')") -> None:
        super(BOEDatasetOCRd, self).__init__()
        self.data = []
        self.text = []
        self.scale = scale
        min_ocr_chars = 500

        self.scale = scale
        self.max_imsize = max_imsize
        

        self.replace = eval(replace_path_expression)            
        
        self.min_width = min_width
        self.min_height = min_height
        self.data = []

        for root, _, files in os.walk(jsons_data_folder):
            for file in tqdm(files, desc=f"Processing dataset..."):
                if not os.path.splitext(file)[1].lower() in ['.json']: continue

                fname = os.path.join(root, file)
                try:
                    datapoint = json.load(open(fname, 'r'))
                except json.JSONDecodeError: continue # TODO: Ensure it happens just a couple of times
                for page in datapoint['pages']:
                    for item in datapoint['pages'][page]:
                        x1, y1, x2, y2 = item['bbox']
                        if ((x2 - x1) > self.min_width) and ((y2 - y1) > self.min_height) and 'ocr' in item:
                            item['ocr'] = item['ocr'].replace('\n', ' ')
                            item['ocr'] = re.sub(r"\s{2,}", " ", item['ocr'])
                            if len(item['ocr']) < min_ocr_chars: continue
                            self.data.append({
                                              'file_uuid': datapoint['file'].replace('.pdf', ''), 
                                              'root': datapoint['path'].replace(*self.replace),
                                               'page': page,
                                               'bbx': item['bbox'],
                                               'text': item['ocr']
                                            })
        print(len(self.data))
        self.device = device
        self.max_crops = 50
        self.tokenizer = None
    
    def iter_text(self):
        for datapoint in self.data: yield datapoint['text']
    
    def __len__(self):
        return len(self.data)
    
    def collate_boe(self, batch):

        max_height = max(crop[0].shape[1] for crop in batch)
        max_width = max(crop[0].shape[2] for crop in batch)

        padded_crops = torch.zeros((len(batch), 3, max_height, max_width))
        supermask = torch.zeros_like(padded_crops)

        embs = []
        for num, (image, emb) in enumerate(batch):

            c, w, h = image.shape

            padded_crops[num, :c, :w, :h] = image
            supermask[num, :c, :w, :h] = 1

            embs += [emb]

        return padded_crops, supermask, torch.stack(embs)
    
    def __getitem__(self, idx):
        
        datapoint = self.data[idx]
        image = read_img(datapoint['root'])[int(datapoint['page'])]

        # resizes
        h, w, _ = image.shape
        new_h, new_w = int(h * self.scale), int(w * self.scale)
        if new_h > new_w and new_h > self.max_imsize:
            ratio = new_w / new_h
            new_h, new_w = self.max_imsize, int(ratio * self.max_imsize)
        
        elif new_h < new_w and new_w > self.max_imsize:

            ratio =  new_h / new_w
            new_h, new_w = int(ratio * self.max_imsize),  self.max_imsize

        image = cv2.resize(image, (new_h, new_w)).transpose(2, 0, 1)
        image = (image - image.mean()) / image.std()
        if self.tokenizer is not None: 
            return image, torch.from_numpy(self.tokenizer.predict(datapoint['text']))
        
        return image, datapoint['text']


