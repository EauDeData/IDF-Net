import torch
import nltk 
import matplotlib.pyplot as plt
import copy

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.dataloaders.dataloaders import DummyDataset
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader
from src.models.models import VisualTransformer
from src.dataloaders.dataloaders import PubLayNetDataset, AbstractsDataset
from src.dataloaders.annoyify import Annoyifier
from src.tasks.evaluation import MAPEvaluation

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
        print('IDF Output:', loader[0])
    except Exception as e:
        print(f"2 - Preprocess test not passed, reason: {e}")
    
    try:
        dataset = DummyDataset()
        cleaner = StringCleanAndTrim()
        loader = LDALoader(dataset, cleaner)
        loader.fit()
        print('LDA Output:', loader[0])
    except Exception as e:
        print(f"3 - Preprocess test not passed, reason: {e}")
    
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

    try:
        data = AbstractsDataset('/home/adria/Desktop/data/arxiv_data.csv', './dataset/arxiv_images')
        print(data[34999][-1])
        plt.imshow(data[34999][0][0])
        plt.show()

    except Exception as e:
        print(f"6 - Dataset test not passed, reason: {e}")
    
    try:
        data = AbstractsDataset('/home/adria/Desktop/data/arxiv_data.csv', './dataset/arxiv_images')
        data.fold = False
        for i in range(len(data)):
            d, t = data[i]
            print(i, end = '\r')

    except Exception as e:
        print(f"7 - Dataset test not passed, reason: {e}")

    try: 

        IMSIZE = 256
        DEVICE = 'cuda'

        ### First we select the dataset ###
        dataset = AbstractsDataset('/home/adri/Downloads/archive/arxiv_data.csv', './dataset/arxiv_images', imsize = IMSIZE)
        print("Tokenizing text!")
        cleaner = StringCleanAndTrim()
        loader = TF_IDFLoader(dataset, StringCleaner())
        loader.fit()
        dataset.tokenizer = loader
        model = VisualTransformer(IMSIZE).eval()
        print('IDF-Vectors with size:', len(dataset[0][1]))
        print('Images with shape', dataset[0][0].shape)
        ann = Annoyifier(dataset, model, 128, len(dataset[0][1]), device = DEVICE, visual='./dataset/visual-LARGE.ann', text='./dataset/text-LARGE.ann')
        print(ann.retrieve_image(dataset[0][0]))

        test_data = copy.deepcopy(dataset)
        test_data.fold = False
        print(MAPEvaluation(test_data, dataset, ann).run())
    
    except Exception as e:
        print(f"8 - Evaluation test not passed, reason: {e}")