import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import torchvision

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, CLIPLoader, BertTextEncoder
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, MSERankLoss
from src.models.models import VisualTransformer, Resnet50, Resnet, ResNetWithEmbedder, AbstractsTopicSpotter
from src.dataloaders.dataloaders import AbstractsDataset, AbstractsAttn
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
DEVICE_BERT = 'cuda:5'
bert = BertTextEncoder().to(DEVICE_BERT)

dataset = AbstractsAttn('train_set.csv', 'dataset/arxiv_images_train/', bert = bert)
dataset_test = AbstractsAttn('test_set.csv', 'dataset/arxiv_images_test/', bert = bert)
del bert
#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
loader = TF_IDFLoader(dataset, cleaner)
loader.fit()

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
#dataset.cleaner = cleaner

dataset_test.tokenizer = loader
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss()
model_visual = ResNetWithEmbedder(embedding_size = 224, resnet = '101') # VisualTransformer(IMSIZE)
model = AbstractsTopicSpotter(model_visual, emb_size = 224, out_size=128,)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = Test(dataset_test, model, loss_function, loader, None, optim, loader, scheduler = scheduler, device = DEVICE, bsize = BSIZE)
train_task = Train(dataset, model, loss_function, loader, None, optim, test_task,loader, device= DEVICE, bsize = BSIZE)

train_task.run(epoches = 120)


