import torch
import nltk 
import matplotlib.pyplot as plt
import pickle
import numpy as np
import clip
from src.text.preprocess import StringCleanAndTrim, StringCleaner
from src.utils.errors import *
from src.text.map_text import LSALoader, TextTokenizer
from src.loss.loss import *
from src.models.models import *
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloaders import BOEDatasetOCRd
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.dataloaders.annoyify import Annoyifier
import seaborn as sns
from sklearn import svm
from tqdm import tqdm
LABELS = {'guerra_ii': 0, 'primo_de_rivera': 1, 'franco': 2, 'republica_ii': 3}


test = BOEDatasetOCRd('/data3fast/users/amolina/BOE/test.txt', scale=1, base_jsons='/data3fast/users/amolina/BOE/', max_imsize=np.inf, acceptance=0, mode='query', resize=None,  min_height = 1, min_width = 1)
hs, ws = list(), list()
for idx in tqdm(range(len(test))):
    img, _, _ = test[idx]
    hs.append(img.shape[0])
    ws.append(img.shape[1])


sns.histplot(hs, kde=True)  # kde=True adds a Kernel Density Estimate plot
plt.xlabel('Height')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.title('Histogram of Matched Regions - Width')  # Title for the plot
plt.savefig('hist1.png')
plt.clf()

sns.histplot(ws, kde=True)  # kde=True adds a Kernel Density Estimate plot
plt.xlabel('Width')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis
plt.title('Histogram of Matched Regions - Height')  # Title for the plot

plt.savefig('hist2.png')