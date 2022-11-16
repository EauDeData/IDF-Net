import torch
import nltk 

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.dataloaders.dataloaders import DummyDataset
from src.text.map_text import LSALoader, TF_IDFLoader
from src.loss.loss import NormLoss, PearsonLoss
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
    

    
