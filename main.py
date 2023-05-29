import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import torchvision
import pickle

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, CLIPLoader
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, MSERankLoss
from src.models.models import VisualTransformer, Resnet50, Resnet, ResNetWithEmbedder
from src.dataloaders.dataloaders import AbstractsDataset, COCODataset
from src.tasks.tasks import Train, Test
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 64
DEVICE = 'cuda:0' # TODO: Implement cuda execution
BSIZE = 64


dataset = AbstractsDataset('train_set.csv', 'dataset/arxiv_images_train/')
dataset_test = AbstractsDataset('test_set.csv', 'dataset/arxiv_images_test/')

#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
#try: 
#loader = pickle.load(open('lda_loader.pkl', 'rb'))
#except:
loader = LSALoader(dataset, cleaner, ntopics = 224)
loader.fit()
#pickle.dump(loader, open('lda_loader.pkl', 'wb'))

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
dataset.twin = False
dataset.cleaner = cleaner

dataset_test.tokenizer = loader
dataset_test.twin = False
dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss(k=1e-2, k_gt=1e-2)
model = Resnet(embedding_size = 64, resnet = '18') # VisualTransformer(IMSIZE)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = Test(dataset, model, loss_function, None, None, optim, bsize = BSIZE, scheduler = False, save = True, device = DEVICE)
train_task = Train(dataset, model, loss_function, None, None, optim, test_task, bsize = BSIZE, device = DEVICE,)

train_task.run(epoches = 120)


