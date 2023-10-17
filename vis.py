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

parser = argparse.ArgumentParser(description="Script for training and testing a model on your dataset.")
parser.add_argument("--name", type=str)
args = parser.parse_args()
name = args.name
if os.path.exists(f'outputs/{name}/failure'): exit()

# https://open.spotify.com/track/0tZYFh6kyraGwBs5VSR3HL?si=5537544b586a46a8

def compute_selfattention(transformer_encoder,x,mask,src_key_padding_mask,i_layer,d_model,num_heads):

    h = F.linear(x, transformer_encoder.layers[i_layer].self_attn.in_proj_weight, bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias)

    h = h.permute(1, 0, 2)

    qkv = h.reshape(x.shape[1], x.shape[0], num_heads, 3 * d_model//num_heads)
    qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
    q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
    d_k = q.size()[-1]
    attn_probs = attn_logits / math.sqrt(d_k)
    # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
    combined_mask = torch.zeros_like(attn_probs)
    if mask is not None:
        combined_mask += mask.float() # assume mask of shape (seq_len,seq_len)
    if src_key_padding_mask is not None:
        combined_mask += src_key_padding_mask.float().unsqueeze(1).unsqueeze(1).repeat(1,num_heads,x.shape[0],1)
        # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
    combined_mask = torch.where(combined_mask>0,torch.zeros_like(combined_mask)-float("inf"),torch.zeros_like(combined_mask))
    # setting masked logits to -inf before softmax
    attn_probs += combined_mask
    attn_probs = F.softmax(attn_probs, dim=-1)
    return attn_logits,attn_probs


def extract_selfattention_maps(transformer_encoder,x,mask,src_key_padding_mask):
    attn_logits_maps = []
    attn_probs_maps = []
    num_layers = transformer_encoder.num_layers
    d_model = transformer_encoder.layers[0].self_attn.embed_dim
    num_heads = transformer_encoder.layers[0].self_attn.num_heads
    norm_first = transformer_encoder.layers[0].norm_first
    with torch.no_grad():
        for i in range(num_layers):
            # compute attention of layer i
            h = x.clone()
            if norm_first:
                h = transformer_encoder.layers[i].norm1(h)
            # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
            # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
            attn_logits,attn_probs = compute_selfattention(transformer_encoder,h,mask,src_key_padding_mask,i,d_model,num_heads)
            attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
            attn_probs_maps.append(attn_probs)
            # forward of layer i
            x = transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
    return attn_logits_maps,attn_probs_maps


# TODO: Test it :-3

class SimilarityToConceptTarget:

    # Source: https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Pixel%20Attribution%20for%20embeddings.ipynb

    def __init__(self, features):
        self.features = features
    
    def __call__(self, model_output):

        if len(model_output.shape) == 1: model_output = model_output.unsqueeze(0)
        if len(self.features.shape) == 1: self.features = self.features.unsqueeze(0)

        cos = torch.nn.CosineSimilarity()
        return cos(model_output, self.features)


class GradCamScanner:
    def __init__(self, model, target, layers, device = 'cuda') -> None:
        self.model = model
        self.target = target
        self.target_layers = layers
        self.use_cuda = device == 'cuda'

    def scan(self, image):
        with GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False) as cam:
            return cam(input_tensor=image[None,], targets=[self.target])[0, :]

class Squarer(torch.nn.Module):
    def __init__(self):
        super(Squarer, self).__init__()
    def forward(self, x):
        return x.view(1, 1, 16, 16)
    
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(1, -1)
    
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

plotable = ['use_topic_image_ViT-B-|-32_lr_1e-05_loss_HardMinerCircle_closs_HardMinerCLR_token_256_accept_0.1_bsize_128_heads_4_layers2_output_256']

# plotable = []
# name = 'use_topic_True_topic_on_image_True_ViT-B-|-32_lr_1e-05_loss_HardMinerCircle_closs_SpearmanRankLoss_token_256_accept_0.4_bsize_128_heads_4_layers2_output_256'

model_clip = clip.load('ViT-B/32', device='cpu')[0].visual

visual = TwoBranchesWrapper(model_clip, 512, 256)
visual.load_state_dict(torch.load(f'{models}/{name}_visual_encoder.pth'))
visual.eval()
visual.to(DEVICE)

pretrained = torch.nn.Sequential(visual.model, visual.contrastive, Squarer(), Flatten())

text_tokenizer = TextTokenizer(StringCleanAndTrim()) 
text_tokenizer.tokens = pickle.load(open(f"{models}/{name}_tokenizer.pth", 'rb'))

cosine_matrix = CosineSimilarityMatrix()

def plot_qualitative(image, text, original_image, pretrained, text_model, text_tokenizer, outname = 'outputs/heats/{:05}.png'.format(idx)):
    pytorch_image = torch.from_numpy(cv2.resize(image, (224, 224)).transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
    concept = pretrained(pytorch_image)[0].to(DEVICE)
    # target_layers = [list(pretrained.modules())[0].model.conv1]
    target_layers = [list(pretrained.modules())[-2]]
    target_layers_tokenizer = [list(pretrained.modules())[2]]

    cam = GradCAM(model=pretrained, target_layers=target_layers, use_cuda=False)
    cam_tokenizer = GradCAM(model=pretrained, target_layers=target_layers_tokenizer, use_cuda=False)

    targets = [SimilarityToConceptTarget(concept)]

    grayscale_cam = cam(input_tensor=torch.from_numpy(cv2.resize(image, (224, 224)).transpose(2, 0, 1)).float().to(DEVICE)[None, ], targets=targets)
    grayscale_cam_tokenizer = cam_tokenizer(input_tensor=torch.from_numpy(cv2.resize(image, (224, 224)).transpose(2, 0, 1)).float().to(DEVICE)[None, ], targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam_tokenizer = grayscale_cam_tokenizer[0, :]


    vis = show_cam_on_image(original_image/255, cv2.resize(grayscale_cam, (image.shape[0], image.shape[1])), use_rgb=True) 
    vis_tokenizer = show_cam_on_image(original_image/255, cv2.resize(grayscale_cam_tokenizer, (image.shape[0], image.shape[1])), use_rgb=True) 

    text_model.load_state_dict(torch.load(f'{models}/{name}_text_encoder.pth'))

    transformer_encoder = text_model.model.encoder
    transformer_encoder.eval()


    tokens = torch.from_numpy(text_tokenizer.predict(text)).int().unsqueeze(1)

    inpuded = text_tokenizer.cleaner(text)

    embs = text_model.model.embedding(tokens.to(DEVICE))

    x = embs

    src_mask = torch.zeros((x.shape[0],x.shape[0])).float().to(DEVICE)

    src_key_padding_mask = torch.zeros((x.shape[1],x.shape[0])).float().to(DEVICE)

    _, attn_probs_maps = extract_selfattention_maps(transformer_encoder,x,src_mask,src_key_padding_mask)

    fig, axs = plt.subplots(len(attn_probs_maps), attn_probs_maps[0].shape[1]+1, figsize = (34, 14))

    cmap = mpl.colormaps.get_cmap('viridis')
    cmap.set_bad("k")
    inpuded = ['<RET>', '<CLS>', '<BOS>'] + inpuded + ['<EOS>'] 
    attn_probs_torch = torch.stack(attn_probs_maps).cpu().numpy()
    min_value, max_value = attn_probs_torch.min(), attn_probs_torch.max()
    for map_idx in range(len(attn_probs_maps)):
        for head_idx in range( attn_probs_maps[0].shape[1]):
            data = attn_probs_maps[map_idx].squeeze()[head_idx].cpu().numpy().round(3) 
            
            data[data < data.mean() + data.std() * .5 ] = 0
            mask = data==0


            if map_idx==1:
                data = data[:, 0]
                mask = mask[:, 0]
                data = data[:, None]
                mask = mask[:, None]
                y_inpuded = inpuded
                x_in = [inpuded[0]]
            else:
                y_inpuded = inpuded
                x_in = inpuded

            sns.heatmap(data, annot = True, mask=mask, yticklabels=y_inpuded, ax = axs[map_idx, head_idx+1], vmin=min_value, vmax=max_value, cmap = cmap, linewidths = 1.2, linecolor = 'white', cbar = False)
            axs[map_idx, head_idx + 1].set_xticklabels(x_in, rotation=45)
            axs[map_idx, head_idx + 1].set_title(f"layer {map_idx}, head {head_idx}")

    colorbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = mpl.colorbar.ColorbarBase(colorbar_ax, cmap=cmap,
                                    norm=mpl.colors.Normalize(vmin=min_value, vmax=max_value))


    axs[0, 0].imshow(vis_tokenizer)
    axs[0, 0].axis('off')
    axs[0, 0].set_title(f"Visual Tokenizer Activations")

    axs[1, 0].imshow(vis)
    axs[1, 0].axis('off')
    axs[1, 0].set_title(f"Visual Encoder Retrieval Token Activations")

    fig.suptitle(f'Softmaxed Attention Maps\n', fontsize = 50)
    plt.savefig(outname, transparent=True)
    plt.show()
    plt.clf()

batchi_batchi = 256
random_samples = [random.randint(0, len(dataset)) for _ in range(batchi_batchi )]

pretrained.to(DEVICE)
pretrained.eval()

images = {}
texts = {}
for i in random_samples:
    im, tex, _ = dataset[i]
    images[i] = visual(torch.from_numpy(cv2.resize(im, (224, 224)).transpose(2, 0, 1)[None,]).float().to(DEVICE))[0].squeeze()
    texts[i] = tex


text_model = TwoBranchesWrapper(TransformerTextEncoder(len(text_tokenizer.tokens), token_size= 256,\
                                                    nheads=4, num_encoder_layers=2),  
                                                    256, 256).to(DEVICE)

text_model.load_state_dict(torch.load(f'{models}/{name}_text_encoder.pth'))
text_model.to(DEVICE)
text_model.eval()

acc_1 = 0
acc_5 = 0
acc_10 = 0
IoU_10 = 0

best = 9999
worst = 0

def plot_idxs(idxs, dataset, name):
    fig, axs = plt.subplots(nrows=1, ncols=len(idxs))
    for n, idx in enumerate(idxs):
        
        datapoint = dataset.data[idx]
        page = datapoint['topic_gt']["page"]
        segment = datapoint['topic_gt']["idx_segment"]
        original_image = np.load(datapoint['root'])[page]
        x, y, x2, y2 = datapoint['pages'][page][segment]['bbox']
        original_image = original_image[y:y2, x:x2]
        
        axs[n].imshow(original_image, interpolation = None)
        axs[n].axis('off')
    fig.tight_layout()
    plt.savefig(name, transparent = True, dpi = 1000)
    plt.clf()
    plt.close(fig)
        
for idx in tqdm(range(len(dataset))):

    with torch.no_grad():
        image, text, _ = dataset[idx]
        query = text_model(torch.from_numpy(text_tokenizer.predict(text)).int().unsqueeze(1).to(DEVICE))[0]

    images_batch = torch.stack([visual(torch.from_numpy(cv2.resize(image, (224, 224)).transpose(2, 0, 1)[None,]).float().to(DEVICE))[0].squeeze()] + [images[i] for i in images if i!=idx])
    text_batch = [text] + [texts[i] for i in texts if i!=idx]
    index_batch = [idx] + [i for i in texts if i!=idx]

    probs = 1 - cosine_matrix(query, images_batch)
    
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
    
    probs_img = 1 - cosine_matrix(images_batch[0].unsqueeze(0), images_batch)
    poss_img = probs_img.argsort().cpu().numpy().tolist()
    IoU_10 += len(set(poss[:11]).intersection(set(poss_img[:11]))) / len(set(poss[:11] + poss_img[:11]))

    if idx>900: continue
    if (name in plotable) and position > worst and position > 10:

        datapoint = dataset.data[idx]
        page = datapoint['topic_gt']["page"]
        segment = datapoint['topic_gt']["idx_segment"]
        original_image = np.load(datapoint['root'])[page]
        x, y, x2, y2 = datapoint['pages'][page][segment]['bbox']
        original_image = original_image[y:y2, x:x2]
        
        dir_ = f'outputs/{name}/failure'
        
        imout = f'{dir_}/{idx:05}_failed_{position}.png'
        imout_fragment = f'{dir_}/{idx:05}_failed_{position}_fragment.png'
        text_out = f'{dir_}/{idx:05}_failed_{position}_text.txt'
        tail = f'{dir_}/{idx:05}_failed_{position}_tail.png'
        
        os.makedirs(dir_, exist_ok=True)
        textos = '\n'.join([f'query: {text}'] + ['retrieved:'] + [text_batch[i].replace('\n', '') for i in poss[:10]])
                
        tail_idxs = [index_batch[i] for i in poss[:5]]
        plot_idxs(tail_idxs, dataset, tail)
        cv2.imwrite(imout_fragment, original_image)
        with open(text_out, 'w') as handler:
            handler.write(textos)
        
        try:
            plot_qualitative(image, text, original_image, pretrained, text_model, text_tokenizer, imout)
        except: pass
    elif (name in plotable) and position < 11:


        datapoint = dataset.data[idx]
        page = datapoint['topic_gt']["page"]
        segment = datapoint['topic_gt']["idx_segment"]
        original_image = np.load(datapoint['root'])[page]
        x, y, x2, y2 = datapoint['pages'][page][segment]['bbox']
        original_image = original_image[y:y2, x:x2]
        
        dir_ = f'outputs/{name}/good'
        
        imout = f'{dir_}/{idx:05}_good_{position}.png'
        imout_fragment = f'{dir_}/{idx:05}_good_{position}_fragment.png'
        text_out = f'{dir_}/{idx:05}_good_{position}_text.txt'
        tail = f'{dir_}/{idx:05}_good_{position}_tail.png'
        
        tail_idxs = [index_batch[i] for i in poss[:5]]
        
        
        os.makedirs(dir_, exist_ok=True)
        plot_idxs(tail_idxs, dataset, tail)
        textos = '\n'.join([f'query: {text}'] + ['retrieved:'] + [text_batch[i].replace('\n', '') for i in poss[:10]])
        
        
        cv2.imwrite(imout_fragment, original_image)
        with open(text_out, 'w') as handler:
            handler.write(textos)
        try:
            plot_qualitative(image, text, original_image, pretrained, text_model, text_tokenizer, imout)
        except: pass
    dir_ = f'outputs/{name}'
    os.makedirs(dir_, exist_ok=True)
    with open(f"{dir_}/results.txt", 'w') as handler:
        handler.write('\nname: ' + name)
        handler.write(f'\n acc_1: {acc_1 / (idx + 1)}')
        handler.write(f'\n acc_5: {acc_5 / (idx + 1)}')
        handler.write(f'\n acc_10: {acc_10 / (idx + 1)}')
        handler.write(f'\n IoU_10: {IoU_10 / (idx + 1)}')
model_clip.cpu()
text_model.cpu()
pretrained.cpu()
torch.cuda.empty_cache()
