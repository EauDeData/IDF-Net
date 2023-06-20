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
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

        image_batch, embs = zip(*batch)
        max_height = max(crop.shape[0] for crop in image_batch)
        max_width = max(crop.shape[1] for crop in image_batch)
        transform = A.Compose([
                A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode = cv2.BORDER_CONSTANT, value = 0),
                ToTensorV2()])

        padded_crops = torch.stack([transform(image = im) for im in image_batch])
        return padded_crops.permute(0, 3, 1, 2), torch.stack(embs)
    
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

        image = cv2.resize(image, (new_h, new_w))
        image = torch.from_numpy((image - image.mean()) / image.std())
        if self.tokenizer is not None: 
            return image, torch.from_numpy(self.tokenizer.predict(datapoint['text']))
        
        return image, datapoint['text']


