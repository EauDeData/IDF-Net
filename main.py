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

### First we select the dataset ###
base = '/home/amolina/Desktop/amolina/COCO/'
transforms = torchvision.transforms.Compose( [torchvision.transforms.Resize((IMSIZE, IMSIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]
)
dataset = COCODataset(f'{base}val2014/', f'{base}captions_val2014.json', transform = transforms)
dataset_test = COCODataset(f'{base}val2014/', f'{base}captions_val2014.json', transform = transforms)

#print(dataset[0][0]['img'].shape)
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
model = ResNetWithEmbedder(embedding_size = 224, resnet = '50') # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = TestMiniClip(dataset_test, model, loss_function, loader, None, optim, loader, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = TrainMiniClip(dataset, model, loss_function, loader, None, optim, test_task,loader, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)


