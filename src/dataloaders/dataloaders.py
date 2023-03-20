import os
from typing import *

import numpy as np
import random
import torch
import json
import cv2
import pandas as pd
from pnglatex import pnglatex
import string
import re

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

class TwinAbstractsDataset:
    name = 'twin_dataset'
    def __init__(self, original_df, imsize, data_folder = 'dataset/augm_images/', level = 'sentence', num = 1, shuffle = True):
        
        self.default_format = \
            r'''
            {title_input}
            

            
            {abstract_input}

            '''
        
        self.dataframe = original_df
        self.separator = '. ' if level == 'sentence' else (' ' if level == 'word' else None)
        self.num = num
        self.imsize = imsize
        self.sh = shuffle
        
        datafolder = datafolder + f"{level}/{num}/{'shuffle' if shuffle else 'no-shuffle'}/"
        if not (os.path.exists(data_folder) and len(os.listdir(data_folder))): self.generate_db(data_folder)
        self.images = datafolder

    def generate_db(self, folder):

        # TODO: Font, salts de linies... Augmentacions exclusives en lo visual.
        printable = set(string.printable)
        print(f"Aux Database not found, generating it on {path}...")
        if not os.path.exists(path): os.mkdir(path)
        for num, (title, abstract) in enumerate(zip(self.dataframe['titles'], self.dataframe['summaries'])):

            print(f"Image number {num}\t", end = '\r')
            sentences = abstract.split(self.separator)

            for _ in range(self.num): sentences.pop(random.randint(0, len(sentences) - 1))
            if self.sh: random.shuffle(sentences)
            abstract = self.separator.join(sentences)

            tex = self.default_format.format(title_input = re.sub(r"[^A-Za-z]+", ' ', title), abstract_input = re.sub(r"[^A-Za-z]+", ' ', abstract))
            pnglatex(tex, f'{path}/{num}.png')

    def __getitem__(self, index):
        
        image = cv2.imread(f"{self.images}/{index}.png")
        image = (image - image.mean()) / image.std()
        image = cv2.resize(image, (self.imsize, self.imsize)).transpose(2, 0, 1)
        image = image.astype(np.float32)

        return image

class AbstractsDataset:

    name = 'abstracts_dataset'
    def __init__(self, csv_path, data_folder, train = True, imsize = 512, twin = False, cleaner = None) -> None:

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
        self.offset = int(.8*len(self.dataframe)) if not train else 0
        self.tokenizer = 0
        self.imsize = imsize
        self.cleaner = cleaner

        self.twin = twin
        if self.twin: self.twin_dataset = TwinAbstractsDataset(self.dataframe, self.imsize)

    def generate_db(self, path) -> None:

        printable = set(string.printable)
        print(f"Database not found, generating it on {path}...")
        if not os.path.exists(path): os.mkdir(path)
        for num, (title, abstract) in enumerate(zip(self.dataframe['titles'], self.dataframe['summaries'])):

            print(f"Image number {num}\t", end = '\r')
            tex = self.default_format.format(title_input = re.sub(r"[^A-Za-z]+", ' ', title), abstract_input = re.sub(r"[^A-Za-z]+", ' ', abstract))
            pnglatex(tex, f'{path}/{num}.png')

        
    def __len__(self):
        if self.fold: return int(.8*len(self.dataframe))
        return int(.2*len(self.dataframe))

    
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
        
        if self.cleaner is not None: text = self.cleaner([text])[0]

        if isinstance(self.tokenizer, int):
            return image, text
        
        if self.twin: return image, self.tokenizer[index], self.twin_dataset[index]
        return image, self.tokenizer.predict(text)

    