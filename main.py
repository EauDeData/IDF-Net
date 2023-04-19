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
from src.models.models import VisualTransformer, Resnet50, Resnet
from src.dataloaders.dataloaders import AbstractsDataset, COCODataset
from src.tasks.tasks import Train, Test, TestMiniClip, TrainMiniClip
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 256
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 16

### First we select the dataset ###
transforms =  torchvision.transforms.Resize((IMSIZE, IMSIZE))
dataset = COCODataset('/home/amolina/Desktop/amolina/COCO/val2014/', '/home/amolina/Desktop/amolina/COCO/captions_val2014.json', transform = transforms)
dataset_test = COCODataset('/home/amolina/Desktop/amolina/COCO/val2014/', '/home/amolina/Desktop/amolina/COCO/captions_val2014.json', transform = transforms)

print(dataset[0])
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
#cleaner = StringCleanAndTrim()
loader = CLIPLoader()
#loader.fit()

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
#dataset.cleaner = cleaner

dataset_test.tokenizer = loader
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = MSERankLoss()
model = Resnet(224, norm = 2, resnet = '152') # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = TestMiniClip(dataset_test, model, loss_function, loader, None, optim, loader, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = TrainMiniClip(dataset, model, loss_function, loader, None, optim, test_task,loader, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)


