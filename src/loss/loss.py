from src.utils.metrics import CosineSimilarityMatrix

import torch
import torch.nn as nn
from typing import *

class CustomLoss:
    def __init__(self) -> None:
        pass
    def __call__(self, h, gt): return self.forward(h, gt)

class NormLoss(CustomLoss):
    def __init__(self, similarity: Callable = CosineSimilarityMatrix(), p_norm: int = 2, margin: float = 0, procrustes = True, *args, **kwargs) -> None:
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
    else: raise NotImplementedError

    mask_diagonal = ~ torch.eye(h_distances.shape[0]).bool()
    h_distances_eyed = h_distances[mask_diagonal].view(gt_distances.shape[0], gt_distances.shape[0]-1) # Shape: (Bs-1*Bs-1)?
    # We remove the diagonal

    gt = gt_distances[mask_diagonal].view(-1)

    # Compute distance and norm
    if orth:
        X = torch.matmul(h_distances_eyed, h_distances_eyed.T)
        V, _, U = torch.linalg.svd(X) # Why X? How do we deal with Non-Square matrices?
        W = torch.matmul(U, V.T)
        sqrd = (torch.matmul(W, h_distances_eyed).view(-1) - gt)**p_norm
    else: sqrd = (h_distances_eyed - gt)**p_norm
    loss = torch.sum(sqrd) ** (1/p_norm) + margin # Limitation of this, you are immitating distances, not topology.
    return loss

def pearson_correlation_loss(features: torch.tensor, gt_distances: torch.tensor, similarity: Callable = CosineSimilarityMatrix(), p_norm: int = 2):

    # From: https://arxiv.org/pdf/2210.05098.pdf
    raise NotImplementedError
    
def minimum_transform_loss(features: torch.tensor, gt: torch.tensor, similarity: Callable = CosineSimilarityMatrix, eye: bool = False) -> torch.tensor:
    '''
    This loss proposes an homeomorphism between spaces.
    
    '''

    # Calculate transformation matrix
    T = torch.matmul(torch.linalg.pinv(features), gt)
    # Ara el loss hauria d'expresar com de "dura" es la matriu de transformació, si és tot 1s genial, si no, problema
    raise NotImplementedError