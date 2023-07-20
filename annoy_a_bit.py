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
from src.utils.metrics import CosineSimilarityMatrix
from src.usage.annoyers import Annoyer
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


resnet_pretrained =  Resnet(embedding_size=224, resnet = '50')
model = torch.nn.Sequential(resnet_pretrained, ProjectionHead(224, 256))
text_model = torch.nn.Sequential(TransformerTextEncoder(len(text_tokenizer.tokens), token_size=224, nheads=8, num_encoder_layers=6), torch.nn.Dropout(p=0.5), ProjectionHead(224, 256)).to(DEVICE)

annoyer = Annoyer(model, text_model, cleaner, text_tokenizer, dataset, 256, 64)
annoyer.fit()

print(annoyer.retrieve_by_idx(0, 1, 0))
