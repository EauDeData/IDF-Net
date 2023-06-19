import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, KullbackDivergenceWrapper
from src.models.models import VisualTransformer, Resnet50
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloaders import BOEDatasetOCRd
from src.tasks.tasks import Train, Test
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 512
DEVICE = 'cpu' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 0.75

### First we select the dataset ###
try:
    dataset = pickle.load(open('boe_fachaset.pkl', 'rb'))
    test_data = pickle.load(open('boe_fachaset_test.pkl', 'rb'))


except FileNotFoundError:
    dataset = BOEDatasetOCRd('/data2fast/users/amolina/train_facha/', max_imsize = IMSIZE)
    test_data = BOEDatasetOCRd('/data2fast/users/amolina/', max_imsize = IMSIZE)

    pickle.dump(dataset, open('boe_fachaset.pkl', 'wb'))
    pickle.dump(test_data, open('boe_fachaset_test.pkl', 'wb'))

print(f"Dataset loader with {len(dataset)} samples...")
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
try: 
    loader = pickle.load(open('lsa_loader.pkl', 'rb'))
except:
    loader = LSALoader(dataset, cleaner, ntopics = 224)
    loader.fit()
    pickle.dump(loader, open('lsa_loader.pkl', 'wb'))

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

