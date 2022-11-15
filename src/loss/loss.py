from src.utils.metrics import CosineSimilarityMatrix

import torch
import torch.nn as nn
from typing import *

class MatchDistanceLoss(nn.Module):
    def __init__(self, similarity: Callable = CosineSimilarityMatrix, p_norm: int = 2, margin: float = 0, *args, **kwargs) -> None:
        super(MatchDistanceLoss).__init__()
        self.similarity = similarity
        self.p_norm = p_norm
        self.margin = margin
    
    def forward(self, h, gt):
        return match_distance_loss(h, gt, self.similarity, self.p_norm, self.margin)

class MatchTopologyLoss(nn.Module):
    pass

def match_distance_loss(features: torch.tensor, gt_distances: torch.tensor, similarity: Callable = CosineSimilarityMatrix, p_norm: int = 2, margin: float = 0) -> torch.tensor:
    '''
    This loss proposes an ismoetry between spaces. We want to measure isomorfism and minimize it.
    Distances1 = Distances2
        features: Torch.tensor: (bs, feature_size)
        gt_distances Torch.tensor: (bs, bs)

    '''
    if isinstance(similarity, CosineSimilarityMatrix):
        h_distances = 1 - similarity(features, features) # Shape: (Bs x Bs)
    else: raise NotImplementedError

    mask_diagonal = ~ torch.eye(h_distances.shape[0]).bool()
    h_distances_eyed = h_distances[mask_diagonal].view(-1) # Shape: (Bs-1*Bs-1)?
    # We remove the diagonal

    gt = gt_distances[mask_diagonal].view(-1)

    # Compute distance and norm
    sqrd = (gt - h_distances_eyed)**p_norm

    loss = sqrd.sum(-1) ** (1/p_norm) + margin # Limitation of this, you are immitating distances, not topology.

    return loss

def minimum_transform_loss(features: torch.tensor, gt: torch.tensor, similarity: Callable = CosineSimilarityMatrix, eye: bool = False) -> torch.tensor:
    '''
    This loss proposes an homeomorphism between spaces.
    
    '''

    # Calculate transformation matrix
    T = torch.matmul(torch.linalg.pinv(features), gt)
    # Ara el loss hauria d'expresar com de "dura" es la matriu de transformació, si és tot 1s genial, si no, problema
    raise NotImplementedError