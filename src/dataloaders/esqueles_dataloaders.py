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
import json

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.dataloaders.base import IDFNetDataLoader

class EsquelaSet:

    name = 'esqueles'
    def __init__(self, jsons_list, base_folder = '/home/amolina/Desktop/datasets/esqueleset/', scale = 0.5,device = 'cuda', max_imsize = 512) -> None:
        super(EsquelaSet, self).__init__()

        # TODO: Test this class
        self.jsons = [json.load(open(os.path.join(base_folder, x.strip()))) for x in open(os.path.join(base_folder, jsons_list), 'r').readlines()]


        self.scale = scale
        self.max_imsize = max_imsize
        self.tokenizer = None
        self.base_path = base_folder
    
    def __len__(self):
        return len(self.jsons)
    
    def iter_text(self):
        for datapoint in self.jsons: yield datapoint['ocr']

    def collate_boe(self, batch):

        image_batch, embs = zip(*batch)
        max_height = max(crop.shape[0] for crop in image_batch)
        max_width = max(crop.shape[1] for crop in image_batch)
        transform = A.Compose([
                A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode = cv2.BORDER_CONSTANT, value = 0),
                ToTensorV2()])

        padded_crops = torch.stack([transform(image = im)['image'] for im in image_batch])
        return padded_crops.float(), torch.stack(embs) 
    
    def __getitem__(self, idx):
        datapoint = self.jsons[idx]
        impath = os.path.join(self.base_path, datapoint['path'])
        image = cv2.imread(impath, cv2.IMREAD_COLOR)

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
        image = (image - image.mean()) / image.std()
        if self.tokenizer is not None:
            return image, torch.from_numpy(self.tokenizer.predict(datapoint['ocr']))
        
        return image, datapoint['ocr']
