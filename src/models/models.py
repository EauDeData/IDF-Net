import torch
from vit_pytorch import ViT
import torch.nn as nn
import torch
import torchvision
from vit_pytorch import ViT
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from src.utils.metrics import CosineSimilarityMatrix
from src.models.attentions import *

class VisualTransformer(torch.nn.Module):
    def __init__(self, image_size, patch_size = 32, embedding_size = 128, depth = 1, heads = 1, dropout = 0.1, norm = 2) -> None:
        super(VisualTransformer, self).__init__()
        self.extractor = ViT(
            image_size = image_size,
            patch_size = patch_size,
            num_classes = 1,
            dim = embedding_size,
            depth = depth,
            heads = heads,
            mlp_dim = 1,
            dropout = dropout,
            emb_dropout = dropout
        )
        self.extractor.mlp_head = list(self.extractor.mlp_head.children())[-2]
        self.norm = norm

    
    def forward(self, batch):
        h =  self.extractor(batch)
        if self.norm is not None: h =  torch.nn.functional.normalize(h, p = self.norm, dim = 1)
        return h


class SelfAttention(torch.nn.Module):
    """ Self attention Layer"""
    # Source: https://discuss.pytorch.org/t/attention-in-image-classification/80147/3
    def __init__(self,in_dim, activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) 

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

class Resnet50(torch.nn.Module):
    def __init__(self, embedding_size = 128, norm = None):
        super(Resnet50, self).__init__()
        self.resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=False)
        self.resnet50.fc = torch.nn.Linear(2048, embedding_size)
        self.norm = norm #lambda x: x if norm is None else lambda x: torch.nn.functional.normalize(x, p = norm, dim = 1)


    def forward(self, batch):

        h = self.resnet50(batch)
        if self.norm is not None: h =  torch.nn.functional.normalize(h, p = self.norm, dim = 1)
        return h
    
class Resnet(torch.nn.Module):
    def __init__(self, embedding_size = 128, norm = None, resnet = '152'):
        super(Resnet, self).__init__()

        if resnet == '152': self.resnet = torchvision.models.resnet152()
        elif resnet == '101': self.resnet =  torchvision.models.resnet101()
        elif resnet == '50': self.resnet =  torchvision.models.resnet50()
        elif resnet == '34': self.resnet =  torchvision.models.resnet34()
        elif resnet == '18': self.resnet =  torchvision.models.resnet18()
        else: raise NotImplementedError


        self.resnet.fc = torch.nn.Linear(2048 if resnet != "18" else 512, embedding_size)
        self.norm = norm
        #self.resnet = torch.nn.DataParallel(self.resnet)


    def forward(self, batch):
        #print(batch.shape)
        h = self.resnet(batch)
        if self.norm is not None: h =  torch.nn.functional.normalize(h, p = self.norm, dim = 1)
        return h
        
def linear_constructor(topology: list):

    seq = []
    for n, size in enumerate(topology[:-1]):
        seq.extend([
            nn.ReLU(),
            nn.Linear(size, topology[n + 1])
        ])
    
    return nn.Sequential(*seq + [nn.LayerNorm(topology[-1])])

def custom_sigmoid(x, t = 1e-3):
    return 1/(1 + torch.exp( -x / t))

class CosinesimilarityAttn(torch.nn.Module):
    def __init__(self, sim = CosineSimilarityMatrix()):
        super(CosinesimilarityAttn, self).__init__()
        self.sim = sim
    
    def forward(self, queries, keys, values):

        attn_weights = self.sim(keys, queries) # (BS_VIS, BS_TEXT)
        #attn_weights = F.softmax(attn_weights, dim = 0) # (BS_VIS, BS_TEXT)

        weighted = torch.matmul(attn_weights.transpose(1, 0), values)
        return weighted, attn_weights
    
class DotProductAttn(torch.nn.Module):

    def __init__(self):
        super(DotProductAttn, self).__init__()
    
    def forward(self, queries, keys, values):

        dot_products = torch.matmul(keys, queries.transpose(1, 0)) # (BS_VIS, BS_TEXT)
        attn_weights = F.softmax(dot_products, dim = 0) # (BS_VIS, BS_TEXT)

        weighted = torch.matmul(attn_weights.transpose(1, 0), values) # For each textual query, a topic model formed with visual information.
        return weighted, attn_weights


class MultiHeadAttn(torch.nn.Module):

    def __init__(self):
        super(MultiHeadAttn, self).__init__()
        self.multihead = MultiHeadAttention() 
    def forward(self, queries, keys, values):

        #dot_products = torch.matmul(keys, queries.transpose(1, 0)) # (BS_VIS, BS_TEXT)
        #attn_weights = F.softmax(dot_products, dim = 0) # (BS_VIS, BS_TEXT)

        #weighted = torch.matmul(attn_weights.transpose(1, 0), values) # For each textual query, a topic model formed with visual information.
        return (x.squeeze() for x in self.multihead(queries.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0)))


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=252,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class AbstractsTopicSpotter(torch.nn.Module):
    def __init__(self, visual_extractor, emb_size, out_size, attn = MultiHeadAttn(), inner_attn = [], bert_size = 768, device = 'cuda') -> None:
        super(AbstractsTopicSpotter, self).__init__()
        self.visual_extractor = visual_extractor
        self.device = device

        hidden_attn = [emb_size] + inner_attn + [out_size]
        self.visual_keys = linear_constructor(hidden_attn)
        self.visual_values = linear_constructor(hidden_attn)
        
        self.textual_queries = linear_constructor([bert_size] + inner_attn + [out_size])
        self.attention_layer = attn
    
    def forward(self, visual_batch, textual_batch, return_values = True):

        visual_features = self.visual_extractor(visual_batch) # SHAPE (BS_VIS, EMB_SIZE)
        visual_values = self.visual_values(visual_features) # SHAPE (BS_VIS, OUT_SIZE)
        visual_keys = self.visual_keys(visual_features) # SHAPE (BS_VIS, OUT_SIZE)

        textual_batch = textual_batch.squeeze()
        textual_queries = self.textual_queries(textual_batch) # SHAPE (BS_TEXT, OUT_SIZE)

        # TODO: Visual keys are not being used, this may cause heavy overfit
        weighted, attn = self.attention_layer(textual_queries, visual_keys, visual_values) # Is visual keys first or textual queries first?

        if return_values: return weighted, visual_values, attn
        return weighted, None, attn
    
class AbstractsMaxPoolTopicSpotter(torch.nn.Module):
    def __init__(self, visual_extractor, emb_size, out_size, inner_attn = [], bert_size = 768, device = 'cuda') -> None:
        super(AbstractsMaxPoolTopicSpotter, self).__init__()
        self.visual_extractor = visual_extractor
        self.device = device

        hidden_attn = [emb_size] + inner_attn + [out_size]
        self.visual_keys = linear_constructor(hidden_attn)
        self.visual_values = linear_constructor(hidden_attn)
        
        self.textual_queries = linear_constructor([bert_size] + inner_attn + [out_size])
    
    def forward(self, visual_batch, textual_batch, return_values = True):

        visual_features = self.visual_extractor(visual_batch) # SHAPE (BS_VIS, EMB_SIZE)
        visual_mean = self.visual_values(visual_features) # SHAPE (BS_VIS, OUT_SIZE)
        visual_variance = self.visual_keys(visual_features) # SHAPE (BS_VIS, OUT_SIZE)

        textual_batch = textual_batch.squeeze()
        textual_queries = self.textual_queries(textual_batch) # SHAPE (BS_TEXT, OUT_SIZE)

        # Objective:
            # For each BS_VIS obtain its respective N textual queries evaluations:
            #
            # mean_1 + query_1 * variance_1
            # mean_1 + query_2 * variance_1
            #       ...
            # Final Shape: [BS_VIS, BS_TEXT, OUT_SIZE]
        # Then take the maxpool for each BS_TEXT
        # Final Shape: [BS_VIS, OUT_SIZE]

        visual_queries_evaluations = visual_mean.unsqueeze(1) + textual_queries.unsqueeze(0) * visual_variance.unsqueeze(1)

        # Then take the maxpool for each BS_TEXT
        # Final Shape: [BS_VIS, OUT_SIZE]
        print(visual_queries_evaluations.shape)
        maxpool, _ = visual_queries_evaluations.max(dim=1)
        print(maxpool.shape)
        exit()

class SimpleEmbedding(torch.nn.Module):
    def __init__(self, in_size, out_size, projector = []):
        super(SimpleEmbedding, self).__init__()
        self.projected = linear_constructor([in_size] + projector + [out_size])

    def forward(self, x):
        return self.projected(x)
