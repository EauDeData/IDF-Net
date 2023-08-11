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
from src.models.models import VisualTransformer, Resnet50, ProjectionHead, Resnet, TransformerTextEncoder
from src.dataloaders.dataloaders import AbstractsDataset
from src.dataloaders.boe_dataloaders import BOEDatasetOCRd
from src.tasks.tasks import Train, Test
from src.tasks.tasks_boe import TrainBOE, TestBOE
from src.tasks.evaluation import MAPEvaluation
from src.utils.metrics import CosineSimilarityMatrix
from src.usage.annoyers import Annoyer
STOPWORDS = nltk.corpus.stopwords.words('spanish')
from src.tasks.visualization import SimilarityToConceptTarget, GradCamScanner, show_cam_on_image

nltk.download('stopwords')
torch.manual_seed(42)

# TODO: Use a config file
# Some constants
IMSIZE = 99999999
DEVICE = 'cuda' # TODO: Implement cuda execution
BSIZE = 64
SCALE = 1
base_jsons = '/data3fast/users/amolina/BOE/'

dataset = BOEDatasetOCRd(base_jsons+'train.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = 224)
test_data = BOEDatasetOCRd(base_jsons+'test.txt', scale = SCALE, base_jsons=base_jsons, max_imsize=IMSIZE,mode='query', resize = 224)

print("Tokenizing text!")
cleaner = StringCleanAndTrim()
try: 
    loader = pickle.load(open('lsa_loader_boe_ok_final.pkl', 'rb'))
except:
    loader = LSALoader(dataset, cleaner, ntopics = 224)
    loader.fit()
    pickle.dump(loader, open('lsa_loader_boe_ok_final.pkl', 'wb'))
try:
    text_tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
except:
    text_tokenizer = TextTokenizer(cleaner)
    text_tokenizer.fit(dataset)
    pickle.dump(loader, open('tokenizer.pkl', 'wb'))

### Now we setup the tokenizer on the dataset ###
dataset.text_tokenizer = text_tokenizer
test_data.text_tokenizer = text_tokenizer

dataset.tokenizer = loader
test_data.tokenizer = loader


# resnet_pretrained =  Resnet(embedding_size=224, resnet = '50')
# model = torch.nn.Sequential(resnet_pretrained, ProjectionHead(224, 256)).to(DEVICE)
pretrained = torch.load('epoch[85]-topic[LSA_mapper]-ntopics[256]-BS[64]-visual_encoder.pkl')
pretrained.eval()
# text_model = torch.nn.Sequential(TransformerTextEncoder(len(text_tokenizer.tokens), token_size=224, nheads=8, num_encoder_layers=6), torch.nn.Dropout(p=0.5), ProjectionHead(224, 256)).to(DEVICE)
text_model = torch.load('epoch[85]-topic[LSA_mapper]-ntopics[256]-BS[64]-text_encoder.pkl')
text_model.eval()
annoyer = Annoyer(pretrained, text_model, cleaner, text_tokenizer, test_data, 256, 224)
annoyer.fit()
# annoyer.load('visual', 'textual', 'topic')

idx = 3217

model = torch.nn.Sequential(pretrained.model, pretrained.contrastive).to(DEVICE)

with torch.no_grad():
    #print(annoyer.retrieve_by_idx(0, 1, 0))
    image, topic, text = test_data[idx]
    print(image.min(), image.max())
    tensor_image = torch.tensor(image.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    target = model(tensor_image).cpu().squeeze().detach()
    nns = annoyer.retrieve_by_vector(target, idx_tree = 1, n = 20)
    encoded_text, _ = text_model(torch.tensor(text).unsqueeze(1).cuda())
    print('=====')
    nns_topic = annoyer.retrieve_by_vector(encoded_text.squeeze(), idx_tree = 0, n = 20)
    nns_topic_lsi = annoyer.retrieve_by_vector(topic, idx_tree = -1, n = 20)

print(test_data.data[idx]['query'])
print('=====')
print([test_data.data[i]['query'] for i in nns[:5]])
print('=====')
print([test_data.data[i]['query'] for i in nns_topic[:5]])
print('=====')
print([test_data.data[i]['query'] for i in nns_topic_lsi[:5]])
print('=====')

print(nns)
print('=====')

print(nns_topic)
print('=====')


print('IoU@50', len(set(nns).intersection(set(nns_topic))) / len(nns + nns_topic))
print('IoU@50', len(set(nns_topic_lsi).intersection(set(nns_topic))) / len(nns_topic_lsi + nns_topic))

print(encoded_text.shape)
print(model)

'''

gradcam_emb = SimilarityToConceptTarget(target.squeeze().to(DEVICE))
gc = GradCamScanner(model, gradcam_emb, list(model.modules())[:1])
gcim = gc.scan(tensor_image.squeeze().to(DEVICE))

print(image, gcim)
vis_query = show_cam_on_image(image, gcim, use_rgb=False,)
cv2.imwrite( 'outputs/tmp_gc_query.png', vis_query)

for i in range(5):
    image_response, _, _ = test_data[nns_topic[i]]
    meta = test_data.data[nns_topic[i]]
    image_response_tensor =  torch.tensor(image_response.transpose(2, 0, 1)).float().unsqueeze(0).cuda()
    gcim = gc.scan(image_response_tensor.squeeze().to(DEVICE))
    #x, y, x2, y2 = meta['pages'][meta['topic_gt']["page"]][meta["topic_gt"]['idx_segment']]['bbox']

    vis_query = show_cam_on_image(image_response, gcim, use_rgb=False, )
    # vis_query = cv2.resize(vis_query, (int(y2 - y), int(x2 - x))[::-1])
    cv2.imwrite( f'outputs/tmp_gc_res_{i}.png', vis_query)


text = ' '.join([test_data.data[i]['query'] for i in range(len(test_data))])
wc = WordCloud(max_words=1000, stopwords=STOPWORDS, margin=10,
               random_state=1).generate(text)
wc.to_file("a_new_hope.png")


'''
