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
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib as mpl
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
random.seed(42)

import argparse



name = 'sentence_bert_and_ocr'
if os.path.exists(f'outputs/{name}/failure'): exit()

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


dataset =  BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = None)

spanish = 'distiluse-base-multilingual-cased-v1'
model = SentenceTransformer(spanish, device = DEVICE).to(DEVICE)
batchi_batchi = 256
random_samples = [random.randint(0, len(dataset)) for _ in range(batchi_batchi )]

images = {}
use_nes= False
for i in random_samples:
    ocr_text = dataset.data[i]['ocr_gt']
    nes = dataset.data[i]['NEs']
    if not use_nes:
        for ne in nes:
            ocr_text = ocr_text.replace(f' {ne[0]} ', ' ')
    
    images[i] = torch.from_numpy(model.encode([ocr_text])).cpu().squeeze()

acc_1 = 0
acc_5 = 0
acc_10 = 0

best = 9999
worst = 0
cosine_matrix = CosineSimilarityMatrix()
for idx in tqdm(range(len(dataset))):


    with torch.no_grad():

        ocr_text = dataset.data[idx]['ocr_gt']
        nes = dataset.data[idx]['NEs']
        text = dataset.data[idx]['query']
        if not use_nes:
            for ne in nes:
                ocr_text = ocr_text.replace(f' {ne[0]} ', ' ')
                text = text.replace(f' {ne[0]} ', ' ')

        query = torch.from_numpy(model.encode([text])).to(DEVICE)
        image =  torch.from_numpy(model.encode([ocr_text])).to(DEVICE).squeeze()


    ocrs_batch = torch.stack( [image] + [images[i].to(DEVICE) for i in images if i!=idx])


    probs = 1 - cosine_matrix(query, ocrs_batch)
    
    poss = probs.argsort().cpu().numpy().tolist()
    position = poss.index(0)

    
    if position == 0:
        acc_1 += 1
        acc_10 += 1
        acc_5 += 1
    elif position < 5:
        acc_10 += 1
        acc_5 += 1
    
    elif position < 10:
        acc_10 += 1

    dir_ = f'outputs/{name}'
    os.makedirs(dir_, exist_ok=True)
    with open(f"{dir_}/results.txt", 'w') as handler:
        handler.write('\nname: ' + name)
        handler.write(f'\n acc_1: {acc_1 / (idx + 1)}')
        handler.write(f'\n acc_5: {acc_5 / (idx + 1)}')
        handler.write(f'\n acc_10: {acc_10 / (idx + 1)}')


