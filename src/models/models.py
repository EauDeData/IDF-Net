import torch
import torchvision
from vit_pytorch import ViT
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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


        self.resnet.fc = torch.nn.Linear(2048, embedding_size)
        self.norm = norm
        #self.resnet = torch.nn.DataParallel(self.resnet)


    def forward(self, batch):
        #print(batch.shape)
        h = self.resnet(batch)
        if self.norm is not None: h =  torch.nn.functional.normalize(h, p = self.norm, dim = 1)
        return h

class ResNetWithEmbedder(torch.nn.Module):
    def __init__(self, resnet='152', pretrained=True, embedding_size: int = 512):
        super(ResNetWithEmbedder, self).__init__()

        if resnet == '152':
            resnet = torchvision.models.resnet152(pretrained=pretrained)
        elif resnet == '101':
            resnet = torchvision.models.resnet101(pretrained=pretrained)
        elif resnet == '50':
            resnet = torchvision.models.resnet50(pretrained=pretrained)
        elif resnet == '34':
            resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif resnet == '18':
            resnet = torchvision.models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError
    
        self.trunk = resnet
        trunk_output_size = self.trunk.fc.in_features
        self.trunk.fc = torch.nn.Identity()
        self.embedder = torch.nn.Linear(trunk_output_size, embedding_size)
        
    def __str__(self):
        return str(self.trunk)

    def forward(self, batch):
        h = self.trunk(batch)
        h = self.embedder(h)
        return h      

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, num_heads=8, hidden_dim=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.transformer_layers = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.mean = nn.Linear(input_dim, output_dim)
        self.variance = nn.Linear(input_dim, output_dim)

    def forward(self, x):

        # Apply transformer layers to padded tensor
        x = self.transformer_layers(x)

        # Take mean across sequence dimension
        x = torch.mean(x, dim=1)

        # Apply linear layer to output global representation
        mu = self.mean(x)
        sigma = self.variance(x)

        return mu, sigma

class DocTopicSpotter(torch.nn.Module):

    def __init__(self, patch_visual_extractor, aggregator, device = 'cuda') -> None:
        super(DocTopicSpotter, self).__init__()
        self.visual_extractor = patch_visual_extractor
        self.aggregator = aggregator
        self.device = device
        self.zeros = torch.zeros(512) # TODO: don't hardcode this

    def forward(self, batch, batch_bert):

        '''

        batch: 
            [{
              "crops": [Tensor[Width, Height], ...]
            },
            {
              "crops": [Tensor[Width, Height], ...]
            }]
        batch_blip: (BATCH_SIZE, BLIP_EMB_SIZE)
        
        '''
        
        # SHAPE: (BATCH, SEQ_LEN_var, EMB_LEN)
        lengths = [len(image['crops']) for image in batch]
        max_len = max(lengths)

        # TODO: Better padding
        visual_tokens = [torch.stack([self.visual_extractor(crop).squeeze() for crop in image['crops']]+[self.zeros] * (lengths[n] - max_len)) for n, image in enumerate(batch)]

        # SHAPE: (BS, EMB)
        input_tensor = torch.stack(visual_tokens)
        mu, sigma = self.aggregator(input_tensor)
        emb = mu + sigma * batch_bert # Text is conditioned by generating a mixture model
        
        return emb
                
