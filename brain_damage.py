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
import random
import clip

import seaborn as sns
import matplotlib as mpl
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
random.seed(42)

import argparse

# TODO: Use a config file
# Some constants
IMSIZE = 99999999
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1

idx = 0
base_jsons = '/data3fast/users/amolina/BOE/'
models = '/data3fast/users/amolina/leviatan'

def compute_similarity(tokens, text_model, visual_vector, tokenizer):
    
    tokens = torch.tensor(tokenizer.predict(tokens)).int().unsqueeze(1).to(DEVICE)
    text_output = text_model(tokens)[0]
    
    return cosine_matrix(text_output, visual_vector).cpu().squeeze().item()


dataset =  BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = 224)
name = 'use_topic_True_topic_on_image_True_ViT-B-|-32_lr_1e-05_loss_HardMinerCircle_closs_HardMinerCircle_token_256_accept_0.4_bsize_128_heads_4_layers2_output_256'

model_clip = clip.load('ViT-B/32', device='cpu')[0].visual

visual = TwoBranchesWrapper(model_clip, 512, 256)
visual.load_state_dict(torch.load(f'{models}/{name}_visual_encoder.pth'))
visual.eval()
visual.to(DEVICE)


text_tokenizer = TextTokenizer(StringCleanAndTrim()) 
text_tokenizer.tokens = pickle.load(open(f"{models}/{name}_tokenizer.pth", 'rb'))

cosine_matrix = CosineSimilarityMatrix()

text_model = TwoBranchesWrapper(TransformerTextEncoder(len(text_tokenizer.tokens), token_size= 256,\
                                                    nheads=4, num_encoder_layers=2),  
                                                    256, 256).to(DEVICE)

text_model.load_state_dict(torch.load(f'{models}/{name}_text_encoder.pth'))
text_model.to(DEVICE)
text_model.eval()

idx = 33
image, text, _ = dataset[idx]
vector = visual(torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE))[0]

base_similarity = compute_similarity(text, text_model, vector, text_tokenizer)

damages = np.zeros((len(text.split()), len(text.split())))
mask = np.tri(len(text.split()),len(text.split()), 0) - np.eye(len(text.split()))
print(mask)
for n, _ in enumerate(text.split()):
    
    for k, _ in enumerate(text.split()):
    
        new_sentence = ' '.join([token if m not in (n, k) else text_tokenizer.unk for m, token in enumerate(text.split())])
        damages[n, k] = base_similarity - compute_similarity(new_sentence, text_model, vector, text_tokenizer)
        print(new_sentence, damages[n, k])

        
plt.figure(figsize=(16, 12))  # Optional: Set the figure size

print(damages.min(), damages.max())
# Use sns.heatmap to create the heatmap
ax = sns.heatmap(damages.round(4), mask = mask, annot=True, cmap = sns.color_palette("icefire", as_cmap=True), linewidths=2, linecolor='white')

# Set the x-axis labels using xticklabels and rotation
ax.set_xticklabels(text.split(), rotation=45, ha='right')

# Set the y-axis labels using yticklabels
ax.set_yticklabels(text.split(), rotation=0)


# Add a title to the plot
plt.title("Damage to the Query - Document Similarity On Masked Tokens (pairs)", fontsize = 20)
plt.savefig(f'damages_paired__masked_{idx}.png', dpi = 100, transparent = True)

damages = np.zeros((len(text.split()), len(text.split())))
for n, _ in enumerate(text.split()):
    
    for k, _ in enumerate(text.split()):
    
        new_sentence = ' '.join([token for m, token in enumerate(text.split()) if m not in (n, k)])
        damages[n, k] = base_similarity - compute_similarity(new_sentence, text_model, vector, text_tokenizer)
        print(new_sentence, damages[n, k])

        
plt.figure(figsize=(16, 12))  # Optional: Set the figure size

print(damages.min(), damages.max())
# Use sns.heatmap to create the heatmap
ax = sns.heatmap(damages.round(4),mask = mask, annot=True, cmap = sns.color_palette("icefire", as_cmap=True), linewidths=2, linecolor='white')

# Set the x-axis labels using xticklabels and rotation
ax.set_xticklabels(text.split(), rotation=45, ha='right')

# Set the y-axis labels using yticklabels
ax.set_yticklabels(text.split(), rotation=0)


# Add a title to the plot
plt.title("Damage to the Query - Document Similarity On Removed Tokens (pairs)", fontsize = 20)
plt.savefig(f'damages_paired__removed_{idx}.png', dpi = 100, transparent = True)


damages = np.zeros((len(text.split()), len(text.split())))
for n, _ in enumerate(text.split()):
    
    for k, _ in enumerate(text.split()):
    
        new_sentence = ' '.join([token if m in (n, k) else text_tokenizer.unk for m, token in enumerate(text.split())])
        damages[n, k] = compute_similarity(new_sentence, text_model, vector, text_tokenizer)
        print(new_sentence, damages[n, k])

        
plt.figure(figsize=(16, 12))  # Optional: Set the figure size

print(damages.min(), damages.max())
# Use sns.heatmap to create the heatmap
ax = sns.heatmap(damages.round(4), annot=True, mask = mask, cmap = sns.color_palette("Blues", as_cmap=True), linewidths=2, linecolor='white')

# Set the x-axis labels using xticklabels and rotation
ax.set_xticklabels(text.split(), rotation=45, ha='right')

# Set the y-axis labels using yticklabels
ax.set_yticklabels(text.split(), rotation=0)

# Add a title to the plot
plt.title("Performance On Masking All Tokens Except Pairs", fontsize = 20)
plt.savefig(f'performance_paired_masked_{idx}.png', dpi = 100, transparent = True)


damages = np.zeros((len(text.split()), len(text.split())))
for n, _ in enumerate(text.split()):
    
    for k, _ in enumerate(text.split()):
    
        new_sentence = ' '.join([token   for m, token in enumerate(text.split()) if m in (n, k) ])
        damages[n, k] = compute_similarity(new_sentence, text_model, vector, text_tokenizer)
        print(new_sentence, damages[n, k])

        
plt.figure(figsize=(16, 12))  # Optional: Set the figure size

print(damages.min(), damages.max())
# Use sns.heatmap to create the heatmap
ax = sns.heatmap(damages.round(4), annot=True, mask = mask, cmap = sns.color_palette("Blues", as_cmap=True), linewidths=2, linecolor='white')

# Set the x-axis labels using xticklabels and rotation
ax.set_xticklabels(text.split(), rotation=45, ha='right')

# Set the y-axis labels using yticklabels
ax.set_yticklabels(text.split(), rotation=0)

# Add a title to the plot
plt.title("Performance On Removing All Tokens Except Pairs", fontsize = 20)
plt.savefig(f'performance_paired__removing_{idx}.png', dpi = 100, transparent = True)