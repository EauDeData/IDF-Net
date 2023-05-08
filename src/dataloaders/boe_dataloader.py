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
        for root, _, files in os.walk(jsons_data_folder):
            for file in tqdm(files, desc=f"Processing dataset..."):
                if not os.path.splitext(file)[1].lower() in ['.json']: continue

                fname = os.path.join(root, file)
                datapoint = json.load(open(fname, 'r'))
                self.data.append(datapoint)

                path = datapoint['path']
                path = os.path.splitext(path)[0]+'.html'
                path = path.replace('images', 'htmls')
                
                sopita = BeautifulSoup(open(path, 'r').read(), features="html.parser")
                self.text.append(sopita.find('h4').text)
        

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
        """
        Custom collate function for the MyDataset class.
        """
        # Get the maximum height and width of the crops in the batch
        max_height = max([sample['crops'].shape[2] for sample in batch])
        max_width = max([sample['crops'].shape[3] for sample in batch])

        # Initialize empty lists for the crops, masks, and textual data
        crops_list = []
        mask_list = []
        textual_list = []
        text_list = []

        # Loop over the batch
        for sample in batch:
            # Append the crops, mask, and textual data to their respective lists
            crops_list.append(sample['crops'])
            mask_list.append(sample['mask'])
            textual_list.append(sample['textual'])
            text_list.append(sample['text'])

        # Stack the crops and mask tensors along the batch dimension and pad to the maximum size
        padded_crops = torch.zeros((len(crops_list), 3, max_height, max_width))
        mask = torch.zeros_like(padded_crops)
        for i, crops in enumerate(crops_list):
            padded_crops[i, :, :crops.shape[2], :crops.shape[3]] = crops
            mask[i, :, :crops.shape[2], :crops.shape[3]] = mask_list[i]

        # Combine the data into a dictionary
        result = {"crops": padded_crops, "mask": mask, "textual": textual_list, "text": text_list}

        return result

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
                
        return {"crops": padded_crops}, {"mask": mask}, textual, self.text[idx]
        


    

