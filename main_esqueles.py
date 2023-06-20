import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle
import numpy as np

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, KullbackDivergenceWrapper
from src.models.models import VisualTransformer, Resnet50
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.esqueles_dataloaders import EsquelaSet
from src.tasks.tasks import Train, Test
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
from src.utils.metrics import CosineSimilarityMatrix
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 224
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1 

### First we select the dataset ###

dataset = EsquelaSet('train.txt', '/home/amolina/Desktop/datasets/esqueleset/', max_imsize = IMSIZE)
test_data = EsquelaSet('test.txt', '/home/amolina/Desktop/datasets/esqueleset', max_imsize = IMSIZE)

print(f"Dataset loader with {len(dataset)} samples...")
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
try: 
    loader = pickle.load(open('lsa_loader_esqueles.pkl', 'rb'))
except:
    loader = LSALoader(dataset, cleaner, ntopics = 224)
    loader.fit()
    pickle.dump(loader, open('lsa_loader_esqueles.pkl', 'wb'))

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
test_data.tokenizer = loader
dataset.scale = SCALE
test_data.scale = SCALE

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss()
model = Resnet50(224, norm = 2) # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = TestBOE(test_data, model, loss_function, loader, cleaner, optim, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = TrainBOE(dataset, model, loss_function, loader, cleaner, optim, test_task, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)

