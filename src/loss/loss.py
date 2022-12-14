from src.utils.metrics import CosineSimilarityMatrix, EuclideanSimilarityMatrix
from src.utils.metrics import mutual_knn

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


class PairwisePotential(CustomLoss):
    def __init__(self, similarity: Callable = EuclideanSimilarityMatrix(), k = 3, mu1 = .5, mu2 = .5, device = 'cuda', *args, **kwargs) -> None:
        self.similarity = similarity
        self.mu1 = mu1
        self.mu2 = mu2
        self.k, self.device = k, device

    def forward(self, h, gt):
        return pairwise_atractors_loss(h, gt, self.similarity, self.k, self.mu1, self.mu2, self.device)

class PearsonLoss(CustomLoss):
    pass

class OrthAligment(CustomLoss):
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

def pairwise_atractors_loss(X, Y, similarity: Callable, k = 3, mu1 = 0.5, mu2 = 0.5, device = 'cuda'):

    #############################################
    # Demonstrated in whiteboard (i think)      #
    # Nn and Nm are mutually close.             #
    # Nx is not.                                #
    # Loss defined by (Nm, Nn, Nx) triplets     #
    # S is a similarity function.               #
    # E = S(Nn, Nx)                             #
    #                                           #
    # Loss = -2sum(atractors) + sum(detractors) #
    #                                           #
    # Loss will be 0 if E is 0                  #
    # and Nm, Nn are 0-similar to Nx            #
    #                                           #
    #############################################

    x_distances = similarity(X, X)
    y_distances = similarity(Y, Y)
    adj_matrix = torch.tensor(mutual_knn(y_distances.cpu().numpy(), k)).to(device)

    atractor = -2 * x_distances * adj_matrix
    repelent = x_distances * abs(adj_matrix - 1)

    L = mu1 * torch.sum(atractor, dim = 0) + mu2 * torch.sum(repelent, dim = 0)

    return torch.sum(L) / L.shape[0]

def clique_potential_loss():
    # The definitive potential loss
    # The thing in the bottom of here, with the scheme bottom-right
    # https://github.com/EauDeData/IDF-Net/blob/main/maths/loss_function_formulation.pdf


    # You will have troubles finding cliques from adj matrix
    # Use this https://www.math.ucdavis.edu/~daddel/linear_algebra_appl/Applications/GraphTheory/GraphTheory_9_17/node10.html
    
    pass

def logprop():

    # Source: https://arxiv.org/pdf/2002.09247.pdf
    # 3.2 Entity - Word Embedding Alignment
    # Probability is proportional to similarity, we can change log P(X, M) ---> Sim(X, M)
    # CONV-augmented model 
    # Same limitation as inequality_satisfied_loss

    pass

def ndcg():

    # We will be shameless and try this again

    pass
