import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import torchvision
import pickle
import multiprocessing as mp

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, CLIPLoader, BertTextEncoder
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, MSERankLoss
from src.models.models import DocTopicSpotter, ResNetWithEmbedder
from src.dataloaders.dataloaders import AbstractsDataset, COCODataset
from src.tasks.tasks import Train, Test, TestMiniClip, TrainMiniClip
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
from src.dataloaders.boe_dataloader import BOEDataset, BOEWhole
from src.tasks.train_doc import TrainDoc


nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 224
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 4 # If batch size is a problem, program properly the collate fn

try:
        dataset = pickle.load(open('output/train.pkl', 'rb'))
        dataset_test = pickle.load(open('output/test.pkl', 'rb'))

except FileNotFoundError:
        dataset = BOEDataset('/home/amolina/Desktop/santa-lucia-dataset/data/train',)
        dataset_test = BOEDataset('/home/amolina/Desktop/santa-lucia-dataset/data/test',)

        pickle.dump(dataset, open('output/train.pkl', 'wb'))
        pickle.dump(dataset_test, open('output/test.pkl', 'wb'))

#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
loader = TF_IDFLoader(dataset, string_preprocess=StringCleanAndTrim())
loader.fit()
scale = 0.25
### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
dataset.scale = scale
#dataset.cleaner = cleaner

dataset_test.tokenizer = loader
dataset_test.scale = scale
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = MSERankLoss()
model = DocTopicSpotter(ResNetWithEmbedder(resnet='50', embedding_size=512), None) # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

trainer = TrainDoc(dataset, dataset_test, model, BertTextEncoder(), loss_function, None, None, optim, None, BSIZE)
trainer.train(100)
