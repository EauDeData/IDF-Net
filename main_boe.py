import torch
import nltk 
import matplotlib.pyplot as plt
import copy
#import wandb
import torchvision
import pickle
import multiprocessing as mp
import cv2
import numpy as np

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
DEVICE = 'cuda:5' # TODO: Implement cuda execution
BERT_DEVICE = 'cuda:4'
BSIZE = 1 # If batch size is a problem, program properly the collate fn
bert =  BertTextEncoder().to(BERT_DEVICE)

try:
        dataset = pickle.load(open('output/train.pkl', 'rb'))
        dataset_test = pickle.load(open('output/test.pkl', 'rb'))

except FileNotFoundError:
        dataset = BOEDataset('/data3fast/users/amolina/BOE/train', bert = bert)
        dataset_test = BOEDataset('/data3fast/users/amolina/BOE/test', bert = bert)

        pickle.dump(dataset, open('output/train.pkl', 'wb'))
        pickle.dump(dataset_test, open('output/test.pkl', 'wb'))

#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
loader = LSALoader(dataset, string_preprocess=StringCleanAndTrim(), ntopics = 128)
loader.fit()
scale = .5
### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
dataset.scale = scale
#dataset.cleaner = cleaner
img = dataset[0][0].numpy()
img = np.hstack(img).transpose(1, 2, 0) * 255
cv2.imwrite('sample_out.png', img.astype(np.uint8))

dataset_test.tokenizer = loader
dataset_test.scale = scale
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss(weighted=None)
emb_size = 128
out_size = 64
model = DocTopicSpotter(ResNetWithEmbedder(resnet='18', embedding_size=emb_size), emb_size, out_size, None) # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 1e-5) # TODO: Don't let it be 1; just for knowing if backprop is properly working
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

model.to(DEVICE)
del bert
trainer = TrainDoc(dataset, dataset_test, model, None, loss_function, None, None, optim, None, BSIZE)
trainer.train(100)
