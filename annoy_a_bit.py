import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle
import cv2
import numpy as np
from wordcloud import WordCloud

from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TF_IDFLoader, LDALoader, TextTokenizer
from src.loss.loss import PairwisePotential, NNCLR, SpearmanRankLoss, KullbackDivergenceWrapper, MSERankLoss, ContrastiveLoss, CLIPLoss, BatchedTripletMargin
from src.models.models import VisualTransformer, Resnet50, ProjectionHead, Resnet, TransformerTextEncoder, TwoBranchesWrapper
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloaders import BOEDatasetOCRd
from src.tasks.tasks import Train, Test
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.utils.metrics import CosineSimilarityMatrix
from src.usage.annoyers import Annoyer
STOPWORDS = nltk.corpus.stopwords.words('spanish')
from src.tasks.visualization import SimilarityToConceptTarget, GradCamScanner, show_cam_on_image
from src.utils.metrics import CosineSimilarityMatrix, get_retrieval_metrics
import random
from tqdm import tqdm
import clip

nltk.download('stopwords')
torch.manual_seed(42)

def get_parameter_from_name(name, parameter):
    return name.split(parameter)[-1].split('_')[-1]


# TODO: Use a config file
# Some constants
IMSIZE = 99999999
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1


base_jsons = '/data3fast/users/amolina/BOE/'
models = '/data3fast/users/amolina/leviatan'
name = 'use_topic_False_topic_on_image_False_ViT-B-|-32_lr_1e-05_loss_SpearmanRankLoss_closs_HardMinerCLR_token_256_accept_0.17_bsize_128_heads_4_layers2_output_256'

text_tokenizer = TextTokenizer(StringCleanAndTrim()) 
text_tokenizer.tokens = pickle.load(open(f"{models}/{name}_tokenizer.pth", 'rb'))


model_clip = clip.load('ViT-B/32', device='cpu')[0].visual
pretrained = TwoBranchesWrapper(model_clip, 512, int(get_parameter_from_name(name, 'output')))
pretrained.load_state_dict(torch.load(f'{models}/{name}_visual_encoder.pth'))
pretrained.eval()
pretrained.to(DEVICE)

text_model = TwoBranchesWrapper(TransformerTextEncoder(len(text_tokenizer.tokens), token_size= 256,\
                                                       nheads=4, num_encoder_layers=2),  
                                                       256, 256).to(DEVICE)

text_model.load_state_dict(torch.load(f'{models}/{name}_text_encoder.pth'))
# dataset = BOEDatasetOCRd(base_jsons+'train.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', acceptance=.17,resize = 224)
test_data = BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = 224)
dataset = None
print("Tokenizing text!")
cleaner = StringCleanAndTrim()
try: 
    loader = pickle.load(open('lsa_loader_boe_ok_final.pkl', 'rb'))
except:
    loader = LSALoader(dataset, cleaner, ntopics = 224)
    loader.fit()
    pickle.dump(loader, open('lsa_loader_boe_ok_final.pkl', 'wb'))

### Now we setup the tokenizer on the dataset ###
test_data.text_tokenizer = text_tokenizer
test_data.tokenizer = loader


annoyer = Annoyer(pretrained, text_model, cleaner, text_tokenizer, test_data, 256, 224)
# annoyer.fit()
annoyer.load('visual', 'textual', 'topic')

#model = pretrained.to(DEVICE)

idx = 505
_, _, text = test_data[idx]
original_text = test_data.data[idx]['query']

print(original_text)

proto_batch = [0, 10476, 1824, 409, 4506, 4012] # [random.randint(0, len(test_data)) for _ in range(128)]
images = [test_data[i][0] for i in proto_batch]
corpus = [test_data[i][2] for i in proto_batch]
texts = [test_data.data[i]['query'] for i in proto_batch]

processed_text = []
processed_images = []
with torch.no_grad():
    for i, (img, original) in tqdm(enumerate(zip(images, corpus))):
        # tokenized = text_tokenizer.predict(original_text)
        # print(tokenized)
        vector = torch.tensor(original).int().unsqueeze(1).to(DEVICE)
        out = text_model(vector)[0].squeeze()
        processed_text.append(out)
        if i == 0: print('tokens', original)

        input_image = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        out_vis = pretrained(input_image)[0].squeeze()
        processed_images.append(out_vis)

c = CosineSimilarityMatrix()
vect = torch.stack([processed_text[0]])
print(vect[0, :5])
a = torch.stack(processed_images)

print(c(vect, a))
#print(get_retrieval_metrics(vect, a))



'''
with torch.no_grad():
    vector = torch.tensor(text).int().unsqueeze(1).to(DEVICE)
    out = text_model(vector)[0].squeeze().cpu().numpy()
    print('Image Annoyer')
    nns = annoyer.retrieve_by_vector(out, n = 5, idx_tree = 0)
    for nn in nns:
        print(nn, '-', test_data.data[nn]['query'])
    
    print('Text Annoyer')
    nns = annoyer.retrieve_by_vector(out, n = 5, idx_tree = 1)
    for nn in nns:
        print(nn, '-', test_data.data[nn]['query'])
'''
