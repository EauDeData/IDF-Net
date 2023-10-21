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
imsize = 224
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

conv_layer = visual.model.conv1
proto_image = torch.ones(1, 3, 512, 512).to(DEVICE)

weights = (conv_layer.weight)

def get_maximum_activation_per_neuron(image, conv_layer, neurons_to_visualize = []):
    stride, ksize = conv_layer.stride[0], conv_layer.kernel_size[0] # Assuming square convolutions
    _, num_channels_out, height, width = (conv_layer(proto_image).shape)
    ouput = conv_layer(image)

    maximum_activated = [None for _ in range(num_channels_out)] 
    for channel in range(num_channels_out):
        
        if len(neurons_to_visualize) and not channel in neurons_to_visualize: continue
        feature = ouput[0, channel]
        feature_height, feature_width = feature.shape

        feature_argmax_1d = feature.view(-1).argmax()
        horizontal_component, vertical_component = feature_argmax_1d % feature_width, feature_argmax_1d // feature_height # Si els resultats són merda segurament sigui perque això està del revès

        crop = image[0, :, vertical_component * stride: (vertical_component * stride) + ksize, horizontal_component * stride: (horizontal_component * stride) + ksize].cpu().numpy().transpose(1,2 , 0)
        
        maximum_activated[channel] = (crop, feature.view(-1)[feature_argmax_1d].item())
    
    return maximum_activated, ouput

def plot_signal(responses, feature_maps, neuron_image, neuron_name):

    sorted_data = sorted(zip(responses, feature_maps), key=lambda x: x[0], reverse=True)
    sorted_responses, sorted_feature_maps = zip(*sorted_data)
    responses = [x / (sum(sorted_responses) - sum(sorted_responses[:n])) for n, x in enumerate(sorted_responses)]


    # Create a barplot for the sorted responses
    ax  = sns.barplot(y=np.array(sorted_responses), x=np.arange(len(sorted_responses)), orient="v")

    # Customize the appearance of the plot
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.set_xticklabels([])  # Remove y-axis labels

    ax.set_xlabel("Responses")
    ax.set_title("Feature Maps vs Responses")

    # Function to display images on top of the bars
    def add_image_to_bar(response, image, ax, idx):
        imagebox = OffsetImage(image, zoom=0.6, resample=True)
        ab = AnnotationBbox(imagebox, (response, idx)[::-1] , frameon=False, pad=0, xycoords="data")
        ax.add_artist(ab)

    # Add feature map images on top of the bars
    for num, (response, image) in enumerate(zip(sorted_responses, sorted_feature_maps)):
        add_image_to_bar(response, image, ax, num)

    os.makedirs(os.path.join(tmp_neurons, neuron_name), exist_ok=True)
    plt.savefig(os.path.join(tmp_neurons, neuron_name, 'activations.png'), dpi = 300)
    plt.clf()
    img = neuron_image.transpose(1,0,2)
    plt.imshow((img - img.min()) / (img.max() - img.min()))
    plt.axis('off')
    plt.savefig(os.path.join(tmp_neurons, neuron_name, 'neuron.png'))
    plt.close()

    return None

# Display the plot

neurons_to_visualize = list(range(0,700, 50))
num_bins = 50
with torch.no_grad():

    all_activations = [[] for _ in range(weights.shape[0])]
    all_signal_responses = [[] for _ in range(weights.shape[0])]
    for idx in tqdm(range(num_bins)):
        try:
            image, text, _ = dataset[idx]
        except: continue # Eliminar quan acabi de possar tot el dataset
        proto_image = torch.from_numpy(cv2.resize(image, (imsize, imsize)).transpose(2, 0, 1)[None,]).float().to(DEVICE)
        activations, output = get_maximum_activation_per_neuron(proto_image, conv_layer, neurons_to_visualize)
        for n,act in enumerate(activations):
            if n in neurons_to_visualize:

                all_activations[n].append(act[0])
                all_signal_responses[n].append(act[1])

    

    ## DO STUFF WITH THE ACTIVATIONS ###
    print(activations)
    activations_stack = []
    labels = []
    for n, activation in enumerate(all_activations):
        if n in neurons_to_visualize:

            responses = all_signal_responses[n]
            activations_normalized = [(img - img.min()) / (img.max() - img.min()) for img,v in zip(activation, responses)]

            fix = plot_signal(responses, activations_normalized, weights[n].cpu().detach().numpy().transpose(1,2, 0), f"channel_{n}_neuron")
