import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, TextTokenizer
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, KullbackDivergenceWrapper, SimCLRLoss
from src.models.models import VisualTransformer, Resnet50
from src.dataloaders.dataloaders import AbstractsDataset
from src.tasks.tasks import Train, Test, TrainCLIPishWithTopic
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
dataset = AbstractsDataset('train_set.csv', 'dataset/arxiv_images_train/', imsize = IMSIZE)
test_data = AbstractsDataset('test_set.csv', 'dataset/arxiv_images_test/', imsize = IMSIZE)

### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim(lang='english')
tokenizer = TextTokenizer(cleaner)
tokenizer.fit(dataset)

loader = LSALoader(dataset, cleaner, ntopics = 224)
loader.fit()

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
test_data.tokenizer = loader

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss()
model = Resnet50(224, norm = 2) # VisualTransformer(IMSIZE)
model_textual = lambda x: x

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = Test(test_data, model, loss_function, loader, cleaner, optim, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = TrainCLIPishWithTopic(dataset, model, model_textual, loss_function, loader, cleaner, optim, test_task, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)

