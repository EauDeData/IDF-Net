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
import matplotlib.pyplot as plt
import torch

from tqdm import tqdm
import albumentations as A
from torch.nn.utils.rnn import pad_sequence
from albumentations.pytorch import ToTensorV2

from src.dataloaders.base import IDFNetDataLoader

def read_img(path):
    img = pdf2image.convert_from_path(path.strip())
    return {str(i): np.array(img[i]) for i in range(len(img))}

class BOEDatasetGraph:

    name = 'boe_dataset'
    def __init__(self, jsons_paths, base_jsons, processor = None, min_nodes = 3, min_height = 125, min_width = 125, min_conf = 0.9, scale = 0.5, device = 'cuda', max_imsize = 64, acceptance = .5, replace_path_expression = "('data1tbsdd', 'data2fast/users/amolina')", mode = 'query', resize = None) -> None:
        super(BOEDatasetGraph, self).__init__()
        self.mode = mode # 'query' or 'ocr'.
        # For test should always be 'query'

        self.scale = scale

        self.scale = scale
        self.max_imsize = max_imsize
        

        self.replace = eval(replace_path_expression)           
        
        self.min_width = min_width
        self.min_height = min_height
        self.data = []
        self.acceptance = acceptance
        for line in open(jsons_paths).readlines():
            path = os.path.join(base_jsons, line.strip()).replace('jsons_gt', 'graphs_gt')
            if os.path.exists(path):
                document = json.load(open(path))
                if document['score'] < acceptance or len(document['graph']['nodes']) < min_nodes: continue
                document['root'] = document['path'].replace(*self.replace).replace('images', 'numpy').replace('.pdf', '.npz')
                page = document['topic_gt']["page"]
                min_x, min_y, max_x, max_y = np.inf, np.inf, 0, 0
        
                for region in document['pages'][page]:
                    if not 'similarity' in region: continue
                    if region['similarity'] >= self.acceptance:
                        x, y, x2, y2 = region['bbox']

                        min_x = min(min_x, x)
                        max_x = max(max_x, x2)
                        
                        
                        min_y = min(min_y, y)
                        max_y = max(max_y, y2)
                
                if (max_x - min_x) >= min_width and (max_y - min_y) >= min_height:
                    self.data.append(document)

        print(len(self.data))
        self.device = device
        self.max_crops = 50
        self.tokenizer = None
        self.graph_tokenizer = None
        self.resize = resize
        self.prop = processor
    
    def iter_text(self):
        for datum in self.data:
            yield datum['query'] if self.mode == 'query' else datum['ocr_gt']
    
    def __len__(self):
        return len(self.data)
    
    def collate_boe(self, batch):

        image_batch, embs, text = zip(*batch)
        max_height = max(crop.shape[0] for crop in image_batch)
        max_width = max(crop.shape[1] for crop in image_batch)
        transform = A.Compose([
                A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode = cv2.BORDER_CONSTANT, value = 0),
                ToTensorV2()])

        padded_crops = torch.stack([transform(image = im)['image'] for im in image_batch])
        text = pad_sequence(text)
        return padded_crops.float(), torch.stack(embs), text

    def get_un_tastet(self, idx):
        image, _, text = self[idx]
        print(text)
        plt.imshow(image)
        plt.title(text)
        plt.savefig(f'tmp_{idx}.png')
        plt.clf()
    
    def __getitem__(self, idx):
        
        datapoint = self.data[idx]
        page = datapoint['topic_gt']["page"]
        image = np.load(datapoint['root'])[page]
        mask = np.zeros_like(image)
        min_x, min_y, max_x, max_y = np.inf, np.inf, 0, 0
        
        for region in datapoint['pages'][page]:
            if not 'similarity' in region: continue
            if region['similarity'] >= self.acceptance:
                x, y, x2, y2 = region['bbox']
                mask[y:y2, x:x2] = 1
                

                min_x = min(min_x, x)
                max_x = max(max_x, x2)
                
                
                min_y = min(min_y, y)
                max_y = max(max_y, y2)


        np_input = (image * mask)
        image = np_input[ min_y:max_y, min_x:max_x, :] 
            
        
        
        # [y:y2, x:x2]

        # resizes
        '''
        
            if self.resize is not None:
            new_h = new_w = self.resize
        else:
            h, w, _ = image.shape
            new_h, new_w = int(h * self.scale), int(w * self.scale)
            new_h, new_w = int(min(new_h, self.max_imsize)), int(min(new_w, self.max_imsize))    
        
                    
        image = cv2.resize(image, (new_h, new_w))
        image = (image - image.mean()) / image.std()
        '''

        
        graph, entities = datapoint['graph'], datapoint['NEs']
        tokens, text = self.graph_tokenizer.predict(graph, entities)
        encoding = self.prop(images=image, text=' '.join(text), padding='max_length', truncation=True, max_length=512, return_tensors="pt") # Don't hardcode this
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding['labels'] = encoding['input_ids']

        return encoding


