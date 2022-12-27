import os
from typing import *

import numpy as np
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

class PubLayNetDataset(IDFNetDataLoader):
    name = 'pubLayNet_dataset'

    '''
    Expected tree:

        publaynet/
                val/*.jpg
                train/*.jpg
                test/*.jpg
                {val, train, test}.json
                README.txt
                LICENSE.txt
    
    src/../dataset/PubLayNetOCR/annot.json
        {
            'gt': [annots...]
            train_ends: index
        }

    '''

    def __init__(self, base_folder: str = '', transcriptions: str = './dataset/PubLayNetOCR/annot.json', ocr: Any = None, train: bool = True, train_p: float = .8, *args, **kwargs) -> None:
        super(PubLayNetDataset).__init__()

        self.train = train
        self.train_p = train_p

        self.data_folder = base_folder if base_folder[-1] == '/' else base_folder+'/'
        self.ocr_path = transcriptions
        self.ocr = ocr(**kwargs)

        self.train_json = json.load(open(base_folder + 'train.json'))
        self.test_json = json.load(open(base_folder + 'test.json'))
        self.val_json = json.load(open(base_folder + 'val.json'))

        self.idToPath = {**{x['id']: x['file_name'] for x in self.train_json['images']},
                         **{x['id']: x['file_name'] for x in self.test_json['images']}, 
                         **{x['id']: x['file_name'] for x in self.val_json['images']}}

        if not os.path.exists(transcriptions):
            try: os.mkdir('./dataset/PubLayNetOCR/')
            except FileExistsError: pass
            self.build_transcriptions(transcriptions)
        else: self.gt = json.load(open(transcriptions, 'r'))
    
    def _total_len(self):
        return len(self.gt['gt']) 

    def __len__(self) -> int:
        '''
        Same class will manage train and test split, therefore we can compute properly the TF-IDF matrix without merging anything.
        '''
        
        if self.train: return(self.gt['train_ends'])
        return(self._total_len() - self.gt['train_ends'])

    def build_transcriptions(self, path):
        self.gt = {
            'gt': [],
            'train_ends': 0
        }
        print(f"Transcriptions not found in {path}; OCRing your database.")
        DOCS_DONE = set()
        def _iter_json(fold, name):
            for element in fold['annotations']:

                # Just Text Category
                if element['category_id'] == 1 and element['id'] in self.idToPath:

                    image = cv2.imread(f"{self.data_folder}{name}/{self.idToPath[element['id']]}", cv2.IMREAD_GRAYSCALE)
                    if not isinstance(image, np.ndarray): continue
                    x, y, w, h = [int(round(u)) for u in element['bbox']] 
                    crop = image[y:y+h, x:x+w]
                    if (not crop.shape[0]*crop.shape[1]) or (element['id'] in DOCS_DONE): continue
                    text = self.ocr.run(crop)['result']
                    print(text)
                    0/0
                    yield {'image': self.idToPath[element['id']], 'bbx': (x, y, w, h), 'text': text}
                    DOCS_DONE.add(element['id'])
        
        for n, element in enumerate(_iter_json(self.train_json, 'train')):
            print(f"OCRing element {n} in train set\t", end = '\r')
            self.gt['gt'].append(element)
        self.gt['train_ends'] = n
        print()
        DOCS_DONE = set()
        # Do we need test? Or is val the fair comparison?
        for n, element in enumerate(_iter_json(self.val_json, 'test')):
            print(f"OCRing element {n} in test set\t", end = '\r')
            self.gt['gt'].append(element)
        print()
        with open(path, 'w', encoding='utf-8') as f:
            print(f"Saving in {path}", end = '\r')
            json.dump(self.gt, f, ensure_ascii=False, indent=4)



    def iter_text(self):
        '''
        No train-test difference, iterate over the wole dataset.
        
        '''
        for element in self.gt['gt']:
            yield element['text']

    def __getitem__(self, index: int) -> Tuple[torch.tensor, str]:
        if not self.train: index = index + len(self)
        item = self.gt['gt'][index]
        impath = f"{self.data_folder}{'train/' if self.train else 'val/'}{item['image']}"
        image = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
        bbx = item['bbx']
        image = image[bbx[1]:bbx[1]+bbx[3], bbx[0]:bbx[0]+bbx[2]]
        return image, item['text']

class AbstractsDataset:

    name = 'abstracts_dataset'
    def __init__(self, csv_path, data_folder, train = True, imsize = 512) -> None:

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
    
    def __getitem__(self, index):
        index = index + self.offset
        image = cv2.imread(f"{self.images}/{index}.png") / 255
        image = cv2.resize(image, (self.imsize, self.imsize)).transpose(2, 0, 1)
        image = image.astype(np.float32)
        text = self.dataframe['titles'][index] + ' ' + \
            self.dataframe['summaries'][index]

        if isinstance(self.tokenizer, int):
            return image, text
        return image, self.tokenizer[index]

    