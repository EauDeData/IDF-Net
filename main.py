import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import torchvision

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, CLIPLoader
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, MSERankLoss
from src.models.models import VisualTransformer, Resnet50, Resnet, ResNetWithEmbedder
from src.dataloaders.dataloaders import AbstractsDataset, COCODataset
from src.tasks.tasks import Train, Test, TestMiniClip, TrainMiniClip
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 224
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64


dataset = AbstractsDataset('train.csv', 'dataset/arxiv_images_train/')
dataset_test = AbstractsDataset('test.csv', 'dataset/arxiv_images_test/')

#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
#cleaner = StringCleanAndTrim()
loader = TF_IDFLoader()
loader.fit()

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
#dataset.cleaner = cleaner

dataset_test.tokenizer = loader
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss()
model = ResNetWithEmbedder(embedding_size = 224, resnet = '101') # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = Test(dataset_test, model, loss_function, loader, None, optim, loader, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = Train(dataset, model, loss_function, loader, None, optim, test_task,loader, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)


