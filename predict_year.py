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
from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib as mpl
import os
import math
import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch.nn as nn
import torch.nn.functional as F
random.seed(42)
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Script for training and testing a model on your dataset.")
parser.add_argument("--name", type=str, default='use_topic_image_ViT-B-|-32_lr_1e-05_loss_HardMinerCircle_closs_HardMinerCLR_token_256_accept_0.1_bsize_128_heads_4_layers2_output_256')
args = parser.parse_args()
name = args.name

# TODO: Use a config file
# Some constants
IMSIZE = 99999999
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1

idx = 0
base_jsons = '/data2/users/amolina/BOE_original/BOEv2/'
models = '/data2/users/amolina/leviatan'

dataset =  BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = None)

tmp_neurons = 'tmp_neurons_receptive'
os.makedirs(tmp_neurons, exist_ok=True)


model_clip = clip.load('ViT-B/32', device='cpu')[0].visual

visual = TwoBranchesWrapper(model_clip, 512, 256)
visual.load_state_dict(torch.load(f'{models}/{name}_visual_encoder.pth'))
visual.eval()
visual.to(DEVICE)

proto_image = torch.rand(1, 3, 224, 224, device = DEVICE)

model_attn = visual.model
conv_block = model_attn.conv1

layer_norm = model_attn.ln_pre
attns_blocks = model_attn.transformer.resblocks.modules

tokens = conv_block(proto_image)
print(tokens)

normalized = layer_norm(tokens)
print(normalized)

print(attns_blocks[0](normalized))