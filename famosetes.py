import torch
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import nltk 
nltk.download('stopwords')

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



import pandas as pd
import json
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
    with torch.no_grad():
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

damages = [['token', 'difference', 'positive', 'query', 'new_query', 'masked']]
nes_freq = {}
for idx in tqdm(range(len(dataset))):
    image, text, _ = dataset[idx]
    
    nes = dataset.data[idx]['NEs']
    vector = visual(torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE))[0]

    base_similarity = compute_similarity(text, text_model, vector, text_tokenizer)
    text = text.replace('\n', ' ').replace(',', ' ').replace('.', '').lower()

    splitted_text = text.split()
    for n, entity_tuple in enumerate(nes):
        entity = entity_tuple[0].lower()

        if not entity in nes_freq: nes_freq[entity] = 0
        nes_freq[entity] += 1
        

        new_sentence = ' '.join([token for token in text.split() if token.lower() != entity])
        damage = base_similarity - compute_similarity(new_sentence, text_model, vector, text_tokenizer)
        if damage == 0: continue # Articles i merdes

        damages.append([nes[n][0], damage, damage > 0, text, new_sentence, False])

        new_sentence = ' '.join([token if token.lower() != entity else '<UNK>' for token in text.split() ])
        damage = base_similarity - compute_similarity(new_sentence, text_model, vector, text_tokenizer)
        
        damages.append([nes[n][0], damage, damage > 0, text, new_sentence, True])

    pd.DataFrame(damages).to_csv('damages.tsv', sep = '\t')
    json.dump(nes_freq, open('nes_freqs.json', 'w'))