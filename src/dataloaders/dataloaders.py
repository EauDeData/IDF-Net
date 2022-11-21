import os
from typing import *

import numpy as np
import torch
import json
import cv2

from src.dataloaders.base import IDFNetDataLoader
from src.text.ocr import TesseractOCR

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

    def __init__(self, base_folder: str = '', transcriptions: str = './dataset/PubLayNetOCR/annot.json', ocr: Any = TesseractOCR, train: bool = True, train_p: float = .8, *args, **kwargs) -> None:
        super(PubLayNetDataset).__init__()

        self.train = train
        self.train_p = train_p

        self.data_folder = base_folder if base_folder[-1] == '/' else base_folder+'/'
        self.ocr_path = transcriptions
        self.ocr = ocr(**kwargs)

        self.train_json = json.load(base_folder + 'train.json')
        self.test_json = json.load(base_folder + 'test.json')
        self.val_json = json.load(base_folder + 'val.json')

        self.idToPath = {**{x['id']: x['file_name'] for x in self.train_json['images']},
                         **{x['id']: x['file_name'] for x in self.test_json['images']}, 
                         **{x['id']: x['file_name'] for x in self.val_json['images']}}

        if not os.path.exists(transcriptions): 
            os.mkdir('./dataset/PubLayNetOCR/')
            self.build_transcriptions(transcriptions)
        else: self.gt = json.load(transcriptions)
    
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
        def _iter_json(fold, name):
            for element in fold['annotations']:

                # Just Text Category
                if element['category_id'] == 1:

                    image = cv2.imread(f"{self.data_folder}{name}/{self.idToPath[element['id']]}", cv2.IMREAD_COLOR)
                    if not isinstance(image, np.ndarray): raise FileNotFoundError
                    x, y, w, h = [int(u) for u in element['bbox']] # TODO: Possible source of conclict, x or y when indexing
                    text = self.ocr.run(image[y:y+h, x:x+w, :])['result']
                    yield {image: self.idToPath[element['id']], 'bbx': (x, y, w, h), 'text': text}
        
        for n, element in enumerate(_iter_json(self.train_json)):
            print(f"OCRing element {n}/{len(n)} in train set\t", end = '\r')
            self.gt['gt'].append(element)
        self.gt['train_ends'] = n
        print()
        # Do we need test? Or is val the fair comparison?
        for n, element in enumerate(_iter_json(self.val_json)):
            print(f"OCRing element {n}/{len(n)} in test set\t", end = '\r')
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

    def __gettitem__(self, index: int) -> Tuple[torch.tensor, str]:
        if not self.train: index_image = index + len(self)
        index_text = index
        raise NotImplementedError
        