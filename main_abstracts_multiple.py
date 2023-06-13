import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle


from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, BertTextEncoder
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, sim_clr_loss, MSERankLoss, KullbackDivergenceWrapper
from src.models.models import VisualTransformer, Resnet50, AbstractsTopicSpotter
from src.models.attentions import ScaledDotProductAttention, AdditiveAttention, DotProductAttention, MultiHeadAttention
from src.dataloaders.dataloaders import AbstractsDataset, AbstractsAttn
from src.tasks.evaluation import MAPEvaluation
from src.tasks.tasks import TrainDocAbstracts
from src.dataloaders.annoyify import Annoyifier
nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 224
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
DEVICE_BERT = 'cuda'
bert = BertTextEncoder().to(DEVICE_BERT)

try: 
    dataset = pickle.load(open('abstracts_dataset.pkl','rb'))
    dataset_test = pickle.load(open('abstracts_dataset_test.pkl','rb'))
except:
    dataset = AbstractsAttn('train_set.csv', 'dataset/arxiv_images_train/', bert = bert,)
    dataset_test = AbstractsAttn('test_set.csv', 'dataset/arxiv_images_test/', bert = bert)

    pickle.dump(dataset, open('abstracts_dataset.pkl','wb'))
    pickle.dump(dataset_test, open('abstracts_dataset_test.pkl','wb'))

dataset.twin = False
dataset_test.twin = False

dataset.imsize = IMSIZE
dataset_test.imsize = IMSIZE
cleaner = StringCleanAndTrim()

del bert
#print(dataset[0][0]['img'].shape)
### On which we clean the text and load the tokenizer ###
print("Tokenizing text!")
try: 
    loader = pickle.load(open('lsa_loader.pkl', 'rb'))
except:
    loader = LSALoader(dataset, cleaner, ntopics = 224)
    loader.fit()
    pickle.dump(loader, open('lsa_loader.pkl', 'wb'))

### Now we setup the tokenizer on the dataset ###
dataset.tokenizer = loader
#dataset.cleaner = cleaner

dataset_test.tokenizer = loader
#dataset_test.cleaner = cleaner

### DL Time: The loss function and model ###
OUT_SIZE = 128
loss_function = SpearmanRankLoss()
model_visual = Resnet50(OUT_SIZE, norm = 2) # VisualTransformer(IMSIZE)
model = AbstractsTopicSpotter(model_visual, emb_size = OUT_SIZE, out_size=512, bert_size=224) # For now we condition with the idf itself

### Optimizer ###
optim = torch.optim.Adam(model.parameters(), lr = 5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min')
closs_function = KullbackDivergenceWrapper()
task = TrainDocAbstracts(dataset, dataset_test, model, None, loss_function, None, None, optim, None, bsize=BSIZE, device='cuda', workers=4, contrastive = closs_function )
task.train(epoches = 120)



