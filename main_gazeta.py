import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle
import numpy as np
import clip

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, TextTokenizer
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, KullbackDivergenceWrapper, MSERankLoss, ContrastiveLoss, CLIPLoss, HardMinerCircle, BatchedTripletMargin, HardMinerTripletLoss, HardMinerCLR
from src.models.models import VisualTransformer, Resnet50, ProjectionHead, Resnet, TransformerTextEncoder, TwoBranchesWrapper
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
IMSIZE = 224
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1
TOKEN_SIZE=64
base_jsons = '/data3fast/users/amolina/BOE/'

dataset = BOEDatasetOCRd(base_jsons+'train.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='ocr', resize=IMSIZE)
test_data = BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='ocr', resize=IMSIZE)
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
loader = LSALoader(dataset, cleaner, ntopics = 256)
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
closs = HardMinerCircle(BSIZE) #HardMinerTripletLoss(BSIZE) # ContrastiveLoss(BSIZE)
loss_function = SpearmanRankLoss() #CLIPLoss() # lambda a,b: torch.nn.functional.cross_entropy(a, b.softmax(dim=1)) # SpearmanRankLoss()
# model = Resnet(embedding_size=256, resnet = '50')
# resnet_pretrained =  Resnet(embedding_size=224, resnet = '50')
# model = torch.nn.Sequential(resnet_pretrained, ProjectionHead(224, 256))
# model = torch.nn.Sequential(VisualTransformer(IMSIZE, patch_size=16, depth=6, heads=8, embedding_size=224), torch.nn.Dropout(p=0), ProjectionHead(224, 256))
model_tag = "ViT-B/32"
model_clip  = clip.load(model_tag, device='cpu')[0].visual
model = TwoBranchesWrapper(model_clip, 512 if 'ViT' in model_tag else 1024, 256)
# model = torch.load('epoch[85]-topic[LSA_mapper]-ntopics[256]-BS[64]-visual_encoder.pkl')
model.train()
text_model = TwoBranchesWrapper(TransformerTextEncoder(len(text_tokenizer.tokens), token_size=TOKEN_SIZE, nheads=2, num_encoder_layers=2), TOKEN_SIZE, 256).to(DEVICE)
#text_model = torch.load('epoch[85]-topic[LSA_mapper]-ntopics[256]-BS[64]-text_encoder.pkl')
text_model.train()
### Optimizer ###

optim = torch.optim.Adam(list(model.parameters()) + list(text_model.parameters()), lr = 1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

test_task = TestBOE(test_data, model, loss_function, loader, cleaner, optim, scheduler = scheduler, device = DEVICE, bsize = BSIZE, text_model=text_model, contrastive_loss=closs)
train_task = TrainBOE(dataset, model, loss_function, loader, cleaner, optim, test_task, device= DEVICE, bsize = BSIZE, text_model=text_model, contrastive_loss=closs)

train_task.run(epoches = 600)

