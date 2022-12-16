from src.utils.metrics import CosineSimilarityMatrix

import torch
import torch.nn as nn
from typing import *
from torch.linalg import norm


class CustomLoss:
    def __init__(self) -> None:
        pass
    def __call__(self, h, gt): return self.forward(h, gt)

class NormLoss(CustomLoss):
    def __init__(self, similarity: Callable = CosineSimilarityMatrix(), p_norm: int = 2, margin: float = 0, procrustes = False, *args, **kwargs) -> None:
        self.similarity = similarity
        self.p_norm = p_norm
        self.margin = margin
        self.orthogonality_solver = procrustes
    
    def forward(self, h, gt):
        return norm_loss(h, gt, self.similarity, self.p_norm, self.margin, self.orthogonality_solver)

class PearsonLoss(CustomLoss):
    def __init__(self,  similarity: Callable = CosineSimilarityMatrix(), p_norm: int = 2) -> None:
        self.similarity = similarity
        self.p_norm = p_norm
    
    def forward(self, h, gt):
        return pearson_correlation_loss(h, gt_distances=gt, similarity=self.similarity, p_norm=self.p_norm)

class OrthAligment(CustomLoss):
    def __init__(self, p = 2) -> None:
        self.p = p

    def forward(self, x, gt):
        return orth_aligment(x, gt, self.p)

class MatchTopologyLoss(nn.Module):
    pass

def norm_loss(features: torch.tensor, gt_distances: torch.tensor, similarity: Callable = CosineSimilarityMatrix(), p_norm: int = 2, margin: float = 0, orth = True) -> torch.tensor:
    '''
    This loss proposes an ismoetry between spaces. We want to measure isomorfism and minimize it.
    Distances1 = Distances2
        features: Torch.tensor: (bs, feature_size)
        gt_distances Torch.tensor: (bs, bs)

    '''
    if similarity.name == 'cosine_matrix':
        h_distances = 1 - similarity(features, features) # Shape: (Bs x Bs)
        gt_distances = 1 - similarity(gt_distances, gt_distances)
    else: raise NotImplementedError

    mask_diagonal = ~torch.eye(h_distances.shape[0]).bool()
    h_distances_eyed = h_distances[mask_diagonal].view(-1) # Shape: (Bs-1*Bs-1)?
    # We remove the diagonal

    gt = gt_distances[mask_diagonal].view(-1)
    

    sqrd = (h_distances_eyed - gt)**p_norm
    loss = torch.sum(sqrd) ** (1/p_norm) + margin # Limitation of this, you are immitating distances, not topology.
    return loss

def pearson_correlation_loss(features: torch.tensor, gt_distances: torch.tensor, similarity: Callable = CosineSimilarityMatrix(), p_norm: int = 2):

    # From: https://arxiv.org/pdf/2210.05098.pdf
    raise NotImplementedError
    
def orth_aligment(features: torch.tensor, target: torch.tensor, p = 2) -> torch.tensor:
    '''
    This loss proposes an isomorphism between spaces.
    
    '''
    # GT Has to be a bunch of vectors
    # TODO: Revisit whats happening here (weird)
    # TODO: Proscrutes doesnt work

    # Calculate transformation matrix
    pinv = torch.linalg.pinv(features)
    R = pinv@target
    dist = (features@R - target) 
    norm = torch.sum(dist ** p, dim=0) ** (1/p)

    return torch.sum(norm)  / features.shape[0] # Normalize for batch size