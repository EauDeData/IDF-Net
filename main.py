import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss
from src.models.models import VisualTransformer, Resnet50
from src.dataloaders.dataloaders import AbstractsDataset
from src.tasks.tasks import Train, Test
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 256
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64

### First we select the dataset ###
dataset = AbstractsDataset('./arxiv_data.csv', './dataset/arxiv_images', imsize = IMSIZE)

### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
loader = LSALoader(dataset, cleaner, ntopics = 224)
loader.fit()

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss(weighted = 'sigmoid')
model = Resnet50(224, norm = 2) # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

### Tasks ###
test_data = copy.deepcopy(dataset)
test_data.fold = False

test_task = Test(test_data, model, loss_function, loader, cleaner, optim, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = Train(dataset, model, loss_function, loader, cleaner, optim, test_task, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)
ann = Annoyifier(dataset, model, 128, len(dataset[0][1]), device = DEVICE, visual='./dataset/visual-[post]-LDA.ann', text='./dataset/text-LDA.ann')
evaluator = MAPEvaluation(test_data, dataset, ann)
res = evaluator.run()
wandb.log(res)
print(res)
