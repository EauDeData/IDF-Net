import torch
import nltk 
import matplotlib.pyplot as plt
import copy

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LSALoaderGLOVE
from src.loss.loss import NormLoss
from src.models.models import VisualTransformer
from src.dataloaders.dataloaders import AbstractsDataset
from src.tasks.tasks import Train, Test
nltk.download('stopwords')

# TODO: Use a config file
# Some constants
IMSIZE = 256
DEVICE = 'cuda' # TODO: Implement cuda execution

### First we select the dataset ###
dataset = AbstractsDataset('/home/adria/Desktop/data/arxiv_data.csv', './dataset/arxiv_images', imsize = IMSIZE)

### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
loader = TF_IDFLoader(dataset, StringCleaner())
loader.fit()

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader

### DL Time: The loss function and model ###
loss_function = NormLoss()
model = VisualTransformer(IMSIZE)

### Optimizer ###
gd = torch.optim.Adam(model.parameters(), lr = 1e-4)
optim = torch.optim.lr_scheduler.ReduceLROnPlateau(gd, 'min')

### Tasks ###
test_data = copy.deepcopy(dataset)
test_data.fold = False

test_task = Test(test_data, model, loss_function, loader, cleaner, optim, device = DEVICE)
train_task = Train(dataset, model, loss_function, loader, cleaner, optim, test_task, device= DEVICE)

train_task.run()

