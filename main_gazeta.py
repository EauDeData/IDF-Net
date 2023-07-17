import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle
import numpy as np

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, TextTokenizer
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, KullbackDivergenceWrapper, MSERankLoss, ContrastiveLoss, CLIPLoss, BatchedTripletMargin
from src.models.models import VisualTransformer, Resnet50, ProjectionHead, Resnet, TransformerTextEncoder
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloaders import BOEDatasetOCRd
from src.tasks.tasks import Train, Test
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
from src.utils.metrics import CosineSimilarityMatrix
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 256
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1
base_jsons = '/data3fast/users/amolina/BOE/'

dataset = BOEDatasetOCRd(base_jsons+'train.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query')
test_data = BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query')
test_data.get_un_tastet(0)
test_data.get_un_tastet(1)
test_data.get_un_tastet(2)

print(f"Dataset loader with {len(dataset)} samples...")
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
#try: 
#    loader = pickle.load(open('lsa_loader_boe_ok.pkl', 'rb'))
#except:
loader = LSALoader(dataset, cleaner, ntopics = 64)
loader.fit()
#    pickle.dump(loader, open('lsa_loader_boe_ok.pkl', 'wb'))
text_tokenizer = TextTokenizer(cleaner)
text_tokenizer.fit(dataset)

### Now we setup the tokenizer on the dataset ###
dataset.text_tokenizer = text_tokenizer
test_data.text_tokenizer = text_tokenizer

dataset.tokenizer = loader
test_data.tokenizer = loader
### DL Time: The loss function and model ###
closs = CLIPLoss()
loss_function = SpearmanRankLoss()
#a = Resnet(embedding_size=224, resnet = '50')
resnet_pretrained =  Resnet(embedding_size=224, resnet = '50')
model = torch.nn.Sequential(resnet_pretrained, ProjectionHead(224, 256))
text_model = torch.nn.Sequential(TransformerTextEncoder(len(text_tokenizer.tokens), token_size=224, nheads=8, num_encoder_layers=6), torch.nn.Dropout(p=0.5), ProjectionHead(224, 256)).to(DEVICE)
### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = TestBOE(test_data, model, loss_function, loader, cleaner, optim, scheduler = scheduler, device = DEVICE, bsize = BSIZE, text_model=text_model, contrastive_loss=closs)
train_task = TrainBOE(dataset, model, None, loader, cleaner, optim, test_task, device= DEVICE, bsize = BSIZE, text_model=text_model, contrastive_loss=closs)

train_task.run(epoches = 420)

