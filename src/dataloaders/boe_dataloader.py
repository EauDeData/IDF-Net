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

from tqdm import tqdm

from src.dataloaders.base import IDFNetDataLoader

def read_img(path):
    img = pdf2image.convert_from_path(path.strip())
    return [np.array(img[i]) for i in range(len(img))]

class BOEDataset:

    name = 'boe_dataset'
    def __init__(self, jsons_data_folder, min_height = 224, min_width = 224, device = 'cuda',) -> None:
        super(BOEDataset, self).__init__()
        self.data = []
        self.text = []
        heading = ['h1', 'h2', 'h3', 'h4']
        max_crops = 15

        for root, _, files in os.walk(jsons_data_folder):
            for file in tqdm(files, desc=f"Processing dataset..."):
                if not os.path.splitext(file)[1].lower() in ['.json']: continue

                fname = os.path.join(root, file)
                datapoint = json.load(open(fname, 'r'))
                total = 0
                for page in datapoint['pages']: total += len(datapoint['pages'][int(page)])
                if total > max_crops: continue

                self.data.append(datapoint)

                path = datapoint['path']
                path = os.path.splitext(path)[0]+'.html'
                path = path.replace('images', 'htmls')
                
                sopita = BeautifulSoup(open(path, 'r').read(), features="html.parser")

                whole_text = []
                for h in heading:
                    whole_text.extend([a.text for a in sopita.find_all(h)])
                self.text.append('\n'.join(whole_text))
        

        self.device = device
        
        self.min_width = min_width
        self.min_height = min_height

        self.tokenizer = 0
        self.max_crops = 50
    
    def iter_text(self):
        for datapoint in self.text: yield datapoint
    
    def __len__(self):
        return len(self.text)
    
    def collate_boe(self, batch):

        max_height = max(crop[0].shape[2] for crop in batch)
        max_width = max(crop[0].shape[3] for crop in batch)
        max_bunch = max(crop[0].shape[0] for crop in batch)

        padded_crops = torch.zeros((len(batch), max_bunch, 3, max_height, max_width))
        supermask = torch.zeros_like(padded_crops)

        texts = []
        embs = []
        for num, (image, mask, emb, text) in enumerate(batch):

            b, c, w, h = image.shape

            padded_crops[num, :b, :c, :w, :h] = image
            supermask[num, :b, :c, :w, :h] = mask

            embs += [emb]
            texts += [text]
        
        return padded_crops, supermask, torch.from_numpy(np.stack(embs)), texts


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
                crops.append(torch.from_numpy(array.transpose(2, 0, 1)).float())

        if not isinstance(self.tokenizer, int): textual = self.tokenizer.predict(self.text[idx])
        else: textual = self.text[idx]
        
        max_height = max(crop.shape[1] for crop in crops)
        max_width = max(crop.shape[2] for crop in crops)

        padded_crops = torch.zeros((len(crops), 3, max_height, max_width))
        mask = torch.zeros_like(padded_crops)

        # TODO: Investigate whats up with the pages with 200 crops
        #    populate output arrays
        for i, crop in enumerate(crops):
            padded_crops[i, :, :crop.shape[1], :crop.shape[2]] = crop
            mask[i, :, :crop.shape[1], :crop.shape[2]] = 1
                
        return padded_crops, mask, textual, self.text[idx]
        

class BOEWhole(BOEDataset):
    def __init__(self, jsons_data_folder, min_height=224, min_width=224, device='cuda') -> None:
        super().__init__(jsons_data_folder, min_height, min_width, device)
        self.just_get = 0
    
    def collate_boe(self, batch):
        
        # padded_crops = torch.zeros((len(batch), 3, max_height, max_width))
        # mask = torch.zeros_like(padded_crops)

        text_emb = []
        texts = []
        batched_inputs = []
        for i, (crop, emb, text) in enumerate(batch):

            batched_inputs.append({"image": crop, "height": crop.shape[1], "width": crop.shape[2]})

            text_emb.append(emb)
            texts.append(text)

        return batched_inputs, torch.from_numpy(np.stack(text_emb)), texts
        
    def __getitem__(self, idx):
        
        image_json = self.data[idx]
        image = read_img(image_json['path'])[self.just_get].transpose(2, 0, 1)
        
        if not isinstance(self.tokenizer, int): textual = self.tokenizer.predict(self.text[idx])
        else: textual = self.text[idx]

        return torch.from_numpy(image), textual, self.text[idx]


    

