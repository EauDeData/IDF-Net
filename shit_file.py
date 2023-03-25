import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import numpy as np
import math
#import wandb

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader
from src.models.models import VisualTransformer, Resnet50, Resnet
from src.dataloaders.dataloaders import AbstractsDataset
from src.tasks.visualization import GradCamScanner, SelfSimilarityTarget, SimilarityToConceptTarget
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 256
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

### First we select the dataset ###
dataset = AbstractsDataset('./arxiv_data.csv', './dataset/arxiv_images', imsize = IMSIZE)

### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
#loader = LSALoader(dataset, cleaner, ntopics = 224)
#loader.fit()   

### Now we setup the tokenizer on the dataset ###
#dataset.tokenizer = loader
data = torch.from_numpy(dataset[9900][0])

### DL Time: The loss function and model ###
#loss_function = SpearmanRankLoss(weighted = 'sigmoid')
model = Resnet(224, resnet = '101') # VisualTransformer(IMSIZE)
model.load_state_dict(torch.load('output/81.pth',))
print('model loaded')

concept = model(data.unsqueeze(0)).squeeze()
print('got the concept')

layers = list(model.modules())[:-5]

scanner = GradCamScanner(model, SimilarityToConceptTarget(concept), layers)

scanned = scanner.scan(data)
scanned = (scanned - scanned.min()) / (scanned.max() - scanned.min())
p = 0.9


plt.imshow(scanned * p + data.squeeze().mean(0).numpy() * (1 - p), cmap = 'gray')
plt.show()
