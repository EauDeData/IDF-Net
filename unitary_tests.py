import torch
import nltk 

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.dataloaders.dataloaders import DummyDataset
from src.text.map_text import LSALoader, TF_IDFLoader
from src.loss.loss import NormLoss, PearsonLoss, OrthAligment
from src.models.models import VisualTransformer
from src.dataloaders.dataloaders import PubLayNetDataset
from src.text.ocr import EasyOCR
nltk.download('stopwords')

if __name__ == '__main__': 
    
    try:
        cleaner_obj = StringCleanAndTrim()
        returned = (cleaner_obj(["I'm Diffie, congrats!", "Hello, why?", "Nick likes to play football, however he is not too fond of tennis."]))
        print(returned)
        if not isinstance(returned, list): raise WrongTypeReturnedeError
    except Exception as e:
        print(f"0 - Preprocess test not passed, reason: {e}") 

    try:
        dataset = DummyDataset()
        cleaner = StringCleanAndTrim()
        loader = LSALoader(dataset, StringCleaner())
        loader.fit()
        print(loader[0])
    except Exception as e:
        print(f"1 - Preprocess test not passed, reason: {e}")
    try:
        dataset = DummyDataset()
        cleaner = StringCleanAndTrim()
        loader = TF_IDFLoader(dataset, cleaner)
        loader.fit()
        print(loader[0])
    except Exception as e:
        print(f"2 - Preprocess test not passed, reason: {e}")

    try:
        loss2 = NormLoss()

        h = torch.rand((5, 10), requires_grad = True)
        gt = torch.ones((5, 5))

        print(loss2(h, gt))
    
    except Exception as e:
        print(f"3 - Loss test not passed, reason: {e}")
    
    try:
        input_tensor = torch.rand((1, 3, 256, 32*2))
        vit = VisualTransformer(256, embedding_size=8)
        print(vit(input_tensor).shape)
    
    except Exception as e:
        print(f"4 - Model (simple ViT) test not passed, reason: {e}")
    
    try:
        h = torch.ones((5, 10), requires_grad = True)
        gt = torch.ones((5, 7), requires_grad = True)
        loss = OrthAligment() 
        print(loss(h, gt))
    except Exception as e:
        print(f"5 - Loss test not passed, reason: {e}")

    dataset = PubLayNetDataset('/home/adria/Desktop/data/publaynet/', ocr=EasyOCR)
    print(dataset.gt['gt'][0])
    print(dataset[0])
    

    
