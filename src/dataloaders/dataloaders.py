import os
from typing import *

import numpy as np
import random
import torch
import torchvision
import json
import cv2
import pandas as pd
from pnglatex import pnglatex
import string
import re
from tqdm import tqdm

from src.dataloaders.base import IDFNetDataLoader

# https://open.spotify.com/track/31i56LZnwE6uSu3exoHjtB?si=1e5e0d5080404042

class DummyDataset(IDFNetDataLoader):
    name = 'dummy_dataset'
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        return 10
    
    def iter_text(self) -> Iterable:
        for _ in range(len(self)): yield "I'm a cat named diffie, my name is not cat but diffie and i like going in the train"

class AbstractsDataset:

    name = 'abstracts_dataset'
    def __init__(self, csv_path, data_folder, train = True, imsize = 512, twin = False) -> None:

        # My Frame https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts?resource=download

        self.dataframe = pd.read_csv(csv_path)

        # TODO: More variate format
        self.default_format = \
            r'''
            {title_input}
            

            
            {abstract_input}

            '''
        
        if not (os.path.exists(data_folder) and len(os.listdir(data_folder))): self.generate_db(data_folder)
        
        self.images = data_folder
        self.fold = train
        self.offset = 0
        self.tokenizer = 0
        self.imsize = imsize

        self.twin = twin
        if self.twin: self.twin_dataset = None

    def generate_db(self, path) -> None:

        printable = set(string.printable)
        print(f"Database not found, generating it on {path}...")
        if not os.path.exists(path): os.mkdir(path)
        for num, (title, abstract) in enumerate(zip(self.dataframe['titles'], self.dataframe['summaries'])):

            print(f"Image number {num}\t", end = '\r')
            tex = self.default_format.format(title_input = re.sub(r"[^A-Za-z]+", ' ', title), abstract_input = re.sub(r"[^A-Za-z]+", ' ', abstract))
            pnglatex(tex, f'{path}/{num}.png')

        
    def __len__(self):
        return len(self.dataframe)

    
    def iter_text(self):
        '''
        No train-test difference, iterate over the wole dataset.
        
        '''
        for num, (title, abstract) in enumerate(zip(self.dataframe['titles'], self.dataframe['summaries'])):

            yield f"{title} {abstract}"

    def get_with_category(self, index):
        
        ret =  self[index]
        index = index + self.offset
        return ret[0], ret[1], eval(self.dataframe['terms'][index])
    
    def get_only_category(self, index):
        index = index + self.offset
        return eval(self.dataframe['terms'][index])
    
    def __getitem__(self, index):
        index = index + self.offset
        image = cv2.imread(f"{self.images}/{index}.png")
        image = (image - image.mean()) / image.std()
        image = cv2.resize(image, (self.imsize, self.imsize)).transpose(2, 0, 1)
        image = image.astype(np.float32)
        text = self.dataframe['titles'][index] + ' ' + \
            self.dataframe['summaries'][index]

        if isinstance(self.tokenizer, int):
            return image, text
        
        if self.twin: return image, self.tokenizer[index], self.twin_dataset[index]
        return image, self.tokenizer.predict(text), text

class AbstractsAttn(AbstractsDataset):
    def __init__(self, csv_path, data_folder, train=True, imsize=512, twin=False, cleaner=None, bert = None) -> None:
        super().__init__(csv_path, data_folder, train, imsize, twin, cleaner)
        self.init_berts(bert)

    def init_berts(self, bert):
        self.berts = []
        self.twin = False
        for title in tqdm(self.dataframe['titles']): 
            with torch.no_grad(): self.berts.append(bert.predict([title]).to('cpu'))
    def collate_boe(self, batch):
        return batch

    def __getitem__(self, index):
        image, topic = super().__getitem__(index)
        return image, topic, self.berts[index]


class COCODataset(torchvision.datasets.CocoCaptions):
    def __init__(self, *args, **kwargs):
        super(COCODataset, self).__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        img, captions = super(COCODataset, self).__getitem__(idx)
        return img, random.choice(captions)
