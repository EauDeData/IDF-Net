import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import torchvision
import pickle

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, CLIPLoader, BertTextEncoder
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, MSERankLoss
from src.models.models import VisualTransformer, Resnet50, Resnet, ResNetWithEmbedder, AbstractsTopicSpotter
from src.dataloaders.dataloaders import AbstractsDataset, AbstractsAttn
from src.tasks.tasks import Train, Test
from src.tasks.train_doc import TrainDocAbstracts
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 224
DEVICE = 'cuda:1' # TODO: Implement cuda execution
BSIZE = 32
DEVICE_BERT = 'cuda:1'
bert = BertTextEncoder().to(DEVICE_BERT)

try: 
    dataset = pickle.load(open('abstracts_dataset.pkl','rb'))
    dataset_test = pickle.load(open('abstracts_dataset_test.pkl','rb'))
except:
    dataset = AbstractsAttn('train_set.csv', 'dataset/arxiv_images_train/', bert = bert,)
    dataset_test = AbstractsAttn('test_set.csv', 'dataset/arxiv_images_test/', bert = bert)

    dataset.init_berts(bert)
    dataset_test.init_berts(bert)

    pickle.dump(dataset, open('abstracts_dataset.pkl','wb'))
    pickle.dump(dataset_test, open('abstracts_dataset_test.pkl','wb'))

dataset.twin = False
dataset_test.twin = False

dataset.imsize = IMSIZE
dataset_test.imsize = IMSIZE
del bert
#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
try: 
    loader = pickle.load(open('lsa_loader.pkl', 'rb'))
except:
    loader = LSALoader(dataset,)
    loader.fit()
    pickle.dump(loader, open('lsa_loader.pkl', 'wb'))
cleaner = StringCleanAndTrim()


### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
#dataset.cleaner = cleaner

dataset_test.tokenizer = loader
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
loss_function = SpearmanRankLoss()
model_visual = ResNetWithEmbedder(embedding_size = 224, resnet = '101') # VisualTransformer(IMSIZE)
model = AbstractsTopicSpotter(model_visual, emb_size = 224, out_size=128, bert_size=64)

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')

task = TrainDocAbstracts(dataset, dataset_test, model, None, loss_function, None, None, optim, None, bsize=BSIZE, device='cuda', workers=3)


task.train(epoches = 120)


