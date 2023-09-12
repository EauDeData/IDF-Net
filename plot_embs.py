import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import nltk 
import matplotlib.pyplot as plt
import copy
import wandb
import pickle
import cv2
import numpy as np
from wordcloud import WordCloud
from PIL import Image
import pandas as pd
from src.text.preprocess import StringCleanAndTrim, StringCleaner
from matplotlib.colors import LinearSegmentedColormap
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
import random
import clip
from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib as mpl
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
random.seed(42)

import argparse

parser = argparse.ArgumentParser(description="Script for training and testing a model on your dataset.")
parser.add_argument("--name", type=str)
args = parser.parse_args()
name = args.name

# https://open.spotify.com/track/0tZYFh6kyraGwBs5VSR3HL?si=5537544b586a46a8

# TODO: Use a config file
# Some constants
IMSIZE = 99999999
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1

idx = 0
base_jsons = '/data3fast/users/amolina/BOE/'
models = '/data3fast/users/amolina/leviatan'
name = 'use_topic_False_topic_on_image_False_ViT-B-|-32_lr_1e-05_loss_SpearmanRankLoss_closs_HardMinerCLR_token_256_accept_0.17_bsize_128_heads_4_layers2_output_256'

model_clip = clip.load('ViT-B/32', device='cpu')[0].visual

visual = TwoBranchesWrapper(model_clip, 512, 256)
visual.load_state_dict(torch.load(f'{models}/{name}_visual_encoder.pth'))
visual.eval()
visual.to(DEVICE)

text_tokenizer = TextTokenizer(StringCleanAndTrim()) 
text_tokenizer.tokens = pickle.load(open(f"{models}/{name}_tokenizer.pth", 'rb'))

text_model = TwoBranchesWrapper(TransformerTextEncoder(len(text_tokenizer.tokens), token_size= 256,\
                                                    nheads=4, num_encoder_layers=2),  
                                                    256, 256).to(DEVICE)


text_model.load_state_dict(torch.load(f'{models}/{name}_text_encoder.pth'))
text_model.to(DEVICE)
text_model.eval()
dataset =  BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = 224)

text_embeddings = []
visual_embedding = []
labels = []
dates = []
ranged = list(range(len(dataset)))
random.shuffle(ranged)
for idx in tqdm(ranged):
    
    image, text, _ = dataset[idx]
    path = dataset.data[idx]['root']
    label = path.split('/')[5]
    dates.append(int(dataset.data[idx]['date'].split('/')[-1]))
    
    numpy_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    text_tensor = torch.tensor(text_tokenizer.predict(text)).int().unsqueeze(1).to(DEVICE)
    
    with torch.no_grad():
        vis_out = visual(numpy_tensor)[0].squeeze().cpu().numpy()
        text_out = text_model(text_tensor)[0].squeeze().cpu().numpy()
    
    text_embeddings.append(text_out)
    visual_embedding.append(vis_out)
    labels.append(label)


colors = {'franco': 'red', 'primo_de_rivera': 'green', 'republica_ii': 'blue', 'guerra_ii': 'purple'}

def plot(data, labels, dates, name, title):
    df = pd.DataFrame(data, columns=['x', 'y'])
    df['label'] = labels
    df['date'] = dates
    cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
    scatter = sns.relplot(data=df, x="x", y="y", hue='date', style="label",  palette=cmap,)
    plt.xlabel('TSNE-X')
    plt.ylabel('TSNE-Y')
    scatter.despine(left=True, bottom=True)
    plt.grid()
    plt.title(title)
    plt.savefig(name, transparent = True, dpi = 505)
    plt.clf()



text_embeddings = np.stack(text_embeddings)
visual_embedding = np.stack(visual_embedding)

text_tsn = TSNE(2).fit_transform(text_embeddings)
vis_tsn = TSNE(2).fit_transform(visual_embedding)

dir_ = f'outputs/{name}'
os.makedirs(dir_, exist_ok=True)
outfile_vis = f"{dir_}/visual_project.png"
outfile_text = f"{dir_}/text_project.png"
plot(text_tsn, labels, dates, outfile_vis, 'Text Embeddings Projection')
plot(vis_tsn, labels, dates, outfile_text, 'Visual Embeddings Projection')

