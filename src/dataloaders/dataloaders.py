import os
from typing import *

import numpy as np
import torch

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
    def __init__(self, base_folder: str = '',transcriptions: str = './dataset/PubLayNetOCR/', ocr: Any = TesseractOCR, train: bool = True, train_p: float = .8, *args, **kwargs) -> None:
        super(PubLayNetDataset).__init__()

        self.train = train
        self.train_p = train_p

        self.data_folder = base_folder
        self.ocr_path = transcriptions
        self.ocr = ocr

        if not os.path.exists(transcriptions): 
            os.mkdir('./dataset/PubLayNetOCR/')
            self.build_transcriptions()
    
    def _total_len(self):
        raise NotImplementedError 

    def __len__(self) -> int:
        '''
        Same class will manage train and test split, therefore we can compute properly the TF-IDF matrix without merging anything.
        '''
        
        if self.train: return(int(self._total_len() * self.train_p))
        return(int(self._total_len * (1 - self.train_p)))

    def build_transcriptions(self):
        raise NotImplementedError

    def iter_text(self):
        '''
        No train-test difference, iterate over the wole dataset.
        
        '''
        raise NotImplementedError

    def __gettitem__(self, index: int) -> Tuple[torch.tensor, str]:
        if not self.train: index_image = index + len(self)
        index_text = index
        raise NotImplementedError
        