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
        
def linear_constructor(topology: list):

    seq = []
    for n, size in enumerate(topology[:-1]):
        seq.extend([
            nn.ReLU(),
            nn.Linear(size, topology[n + 1])
        ])
    
    return nn.Sequential(*seq)

def custom_sigmoid(x, t = 1e-3):
    return 1/(1 + torch.exp( -x / t))

class AbstractsTopicSpotter(torch.nn.Module):
    def __init__(self, visual_extractor, emb_size, out_size, attn, inner_attn = [], bert_size = 768, device = 'cuda') -> None:
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
        # visual_keys = self.visual_keys(visual_features) # SHAPE (BS_VIS, OUT_SIZE)

        textual_batch = textual_batch.squeeze()
        textual_queries = self.textual_queries(textual_batch) # SHAPE (BS_TEXT, OUT_SIZE)

        dim = 0

        # TODO: Visual keys are not being used, this may cause heavy overfit
        weighted, attn = self.attention_layer(textual_queries.unsqueeze(dim), visual_values.unsqueeze(dim)) # Is visual keys first or textual queries first?
        weighted, attn = weighted.squeeze(), attn.squeeze()        

        if return_values: return weighted, visual_values
        return weighted, None
    
class AbstractsMaxPoolTopicSpotter(torch.nn.Module):
    def __init__(self, visual_extractor, emb_size, out_size, attn, inner_attn = [], bert_size = 768, device = 'cuda') -> None:
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

        return None, None