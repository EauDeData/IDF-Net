import torch
import nltk 
import matplotlib.pyplot as plt
import copy
from scipy import stats
import torch.utils.data.dataloader as dataloader
import pickle

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.dataloaders.dataloaders import DummyDataset, COCODataset
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, BertTextEncoder
from src.models.models import DocTopicSpotter, ResNetWithEmbedder, TransformerEncoder, YOTARO, DocTopicSpotter, AbstractsTopicSpotter
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloader import BOEDataset, BOEWhole, read_img

from src.dataloaders.annoyify import Annoyifier
from src.tasks.evaluation import MAPEvaluation
from src.loss.loss import (nns_loss, rank_correlation, rank_correlation_loss, CosineSimilarityMatrix,
                           smooth_rank, sigmoid, batched_spearman_rank, corrcoef)


nltk.download('stopwords')

if __name__ == '__main__': 

    model = AbstractsTopicSpotter(ResNetWithEmbedder(resnet = '18', ), 512, 128)
    images = torch.zeros(8, 3, 224, 224)
    text = torch.zeros(8, 768)
    print(model(images, text).shape)
    exit()
    try:
        a, b = torch.rand(5, 5), torch.rand(5, 15)
        sim_a = CosineSimilarityMatrix()(a, a)
        sim_b = CosineSimilarityMatrix()(b, b)

        rank_a = smooth_rank(sim_a, 1e-5, sigmoid)
        rank_b = smooth_rank(sim_b, 1e-5, sigmoid)

        print(sum(corrcoef(rank_a, rank_b))/5)
        z, y = batched_spearman_rank(rank_a, rank_b)
        print(sum(z) / len(z))
        print(rank_correlation_loss(a, b))
    except: pass



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
        print('predicted:, ',loader.predict('cat'))
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
        dataset = DummyDataset()
        cleaner = StringCleanAndTrim()
        loader = LSALoader(dataset, cleaner)
        loader.fit()
        print('LSA Output:', loader[0])
    except Exception as e:
        print(f"3.5 - Preprocess test not passed, reason: {e}")
    
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
