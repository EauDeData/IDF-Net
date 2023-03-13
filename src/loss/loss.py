from src.utils.metrics import CosineSimilarityMatrix, EuclideanSimilarityMatrix, EuclideanDistanceMatrix
from src.utils.metrics import mutual_knn, knn, batched_spearman_rank, sigmoid, corrcoef, cov

import torch
import torch.nn as nn
from typing import *
from torch.linalg import norm
import numpy as np

class CustomLoss:
    def __init__(self) -> None:
        pass
    def __call__(self, h, gt): return self.forward(h, gt)

class PairwisePotential(CustomLoss):
    def __init__(self, similarity: Callable = EuclideanSimilarityMatrix(), k = 3, mu1 = .5, mu2 = .5, device = 'cuda', *args, **kwargs) -> None:
        self.similarity = similarity
        self.mu1 = mu1
        self.mu2 = mu2
        self.k, self.device = k, device

    def forward(self, h, gt):
        return pairwise_atractors_loss(h, gt, self.similarity, self.k, self.mu1, self.mu2, self.device)

class NNCLR(CustomLoss):
    def __init__(self, similarity = EuclideanDistanceMatrix()):
        self.similarity = similarity
    
    def forward(self, h, gt):
        return nns_loss(h, gt, self.similarity)
    
class SpearmanRankLoss(CustomLoss):
    def __init__(self, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5, device = 'cuda', weighted = None, maxy = 3):

        self.ind = indicator_function
        self.sim = similarity
        self.scale = scale
        self.k = k 
        self.k_gt = k_gt
        self.weight = weighted
        self.device = device
        self.maxy = maxy

    def forward(self, h, gt):
        return rank_correlation_loss(h, gt, self.ind, self.sim, self.scale, self.k, self.k_gt, self.weight, self.maxy, self.device)


class MSERankLoss(CustomLoss):
    def __init__(self, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5, device = 'cuda', weighted = None, maxy = 3):

        self.ind = indicator_function
        self.sim = similarity
        self.scale = scale
        self.k = k 
        self.k_gt = k_gt
        self.weight = weighted
        self.device = device
        self.maxy = maxy

    def forward(self, h, gt):
        return mse_rank_loss(h, gt, self.ind, self.sim, self.scale, self.k, self.k_gt)

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

def sim_clr_loss(h, h_augm, temperature = 0.1):
    pass

def nns_loss(h, gt, distance_function = EuclideanDistanceMatrix(), temperature = 0.1):
    # From " With a Little Help from My Friends" paper (insptiration)
    n = h.shape[0]

    distances = distance_function(gt, gt)
    friends = knn(distances) * 1
    eyed = torch.eye(n, dtype = bool)

    closest = h[torch.argmax(friends, dim = 1)]
    diag = h[torch.argmax(eyed * 1, dim = 1)]

    numerator = torch.bmm(closest[:, None, :], diag[:, :, None]).view(-1)
    numerator = (numerator/temperature).exp()
    
    denominator = torch.matmul(h, h.T)[~eyed].view(n, n-1)
    denominator = (denominator/temperature).exp().sum(dim = 1)
    
    nn_clr = -torch.log(numerator / denominator)

    return nn_clr.mean()

def smooth_rank(sm, temperature, indicator_function):
    mask_diagonal = ~ torch.eye(sm.shape[0]).bool()
    ranking = sm[mask_diagonal].view(sm.shape[0], sm.shape[0]-1)

    # Prepare indicator function
    dij = ranking.unsqueeze(1) - ranking.unsqueeze(-1)
    mask_diagonal = ~ torch.eye(dij.shape[-1]).bool()
    dij = dij[:,mask_diagonal].view(dij.shape[0], dij.shape[1], -1)

    # Smooth indicator function
    indicator = indicator_function(dij, k=temperature)
    indicator = indicator.sum(-1) + 1
    return sm.shape[0] - indicator


def rank_correlation_loss(h, target, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5, weighting = 'sigmoid', maxy = 3, device = 'cuda'):

    sm = similarity(h, h)
    indicator = smooth_rank(sm, k, indicator_function)
    
    # Ground-truth Ranking function
    gt_sim = similarity(target, target)
    gt_indicator = smooth_rank(gt_sim, k_gt, indicator_function)

    n = h.shape[0] if scale else 1
    C = corrcoef(gt_indicator, indicator)
    if weighting is not None:

        bins = torch.flip((torch.arange(h.shape[0] - 1).to(device) + 1), [0]) 
        delta = bins *  maxy / (h.shape[0] - 1)        
        if weighting == 'sigmoid':

            W = 1 / (1 + torch.exp(delta))

        Idiff = torch.abs(indicator - gt_indicator) * W.unsqueeze(0)
        scalator = Idiff.sum(1) / W.sum() + 1
        scalator.detach()
    
    else:
        scalator = 1

    return 1 - (torch.sum(C * scalator/ n))

def mse_rank_loss(h, target, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5,):

    sm = similarity(h, h)
    indicator = smooth_rank(sm, k, indicator_function)
    
    # Ground-truth Ranking function
    gt_sim = similarity(target, target)
    gt_indicator = smooth_rank(gt_sim, k_gt, indicator_function)

    n = h.shape[0] if scale else 1

    return torch.mean(torch.pow(indicator - gt_indicator, 2).sum(1)) / (n ** 2)

### Evaluation Metrics ###
def rank_correlation(a, b, distance_function = CosineSimilarityMatrix()):
    
    sim_a = CosineSimilarityMatrix()(a, a)
    sim_b = CosineSimilarityMatrix()(b, b)

    rank_a = smooth_rank(sim_a, 1e-5, sigmoid)
    rank_b = smooth_rank(sim_b, 1e-5, sigmoid)

    statistics, pvalues = batched_spearman_rank(rank_a, rank_b)

    return np.mean(statistics), np.mean(pvalues)


def raw_accuracy(h, gt, distance_function = EuclideanDistanceMatrix()):
    pass
