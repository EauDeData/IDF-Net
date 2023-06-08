from typing import Any
import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import torchvision
import pickle

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, MSERankLoss
from src.models.models import VisualTransformer, Resnet50, Resnet
from src.dataloaders.dataloaders import AbstractsDataset
from src.tasks.tasks import Train, Test
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 128
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 21


dataset = AbstractsDataset('train_set.csv', 'dataset/arxiv_images_train/')
dataset_test = AbstractsDataset('test_set.csv', 'dataset/arxiv_images_test/')

#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
class ProxyCleaner:
    def __init__(self) -> None:
        pass

    def __call__(self, batch, *args: Any, **kwds: Any) -> Any:
        return batch.split()
try: 
    loader = pickle.load(open('lda_loader.pkl', 'rb'))

except:
    loader = LDALoader(dataset, cleaner, num_topics=32)
    loader.fit()
    pickle.dump(loader, open('lda_loader.pkl', 'wb'))

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
dataset.twin = False

dataset_test.tokenizer = loader
dataset_test.twin = False

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss()
model = Resnet(embedding_size = 64, resnet = '50').to(DEVICE) # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

####### DEBUG REGION ######
from src.utils.metrics import CosineSimilarityMatrix
import matplotlib.pyplot as plt
import numpy as np

matrix = CosineSimilarityMatrix()
dataloader = torch.utils.data.DataLoader(dataset, batch_size = BSIZE)
for out  in dataloader:
    try:
        samples, topics, text = out
    except:
        samples, topics = out
    print("input shape", samples.shape)
    print("matriu (ara):")
    print(matrix(topics, topics))
    samples = samples.to(DEVICE)
    topics = topics.to(DEVICE)
    features = model(samples)

    print("matriu (bé):")
    lloss = loss_function(features, topics)
    lloss.backward()

    optim.step()
    optim.zero_grad()
    print(lloss)
    if lloss!=lloss: break


