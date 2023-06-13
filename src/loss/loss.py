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
    def __init__(self, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5):

        self.ind = indicator_function
        self.sim = similarity
        self.scale = scale
        self.k = k 
        self.k_gt = k_gt

    def forward(self, h, gt):
        return rank_correlation_loss(h, gt, self.ind, self.sim, self.scale, self.k, self.k_gt)
    

class SimCLRLoss(CustomLoss):
    def __init__(self, similarity = CosineSimilarityMatrix(), k = .5) -> None:

        self.sim = similarity
        self.t = k

    def forward(self, h, gt):
        return sim_clr_loss(h, gt, self.t, self.sim)

class MSERankLoss(CustomLoss):
    def __init__(self, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5):

        self.ind = indicator_function
        self.sim = similarity
        self.scale = scale
        self.k = k 
        self.k_gt = k_gt

    def forward(self, h, gt):
        return mse_rank_loss(h, gt, self.ind, self.sim, self.scale, self.k, self.k_gt)

class KullbackDivergenceWrapper(CustomLoss):
    def __init__(self) -> None:

        self.loss = torch.nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, h, gt):

        h = h / torch.sum(h, 1)
        gt = gt / torch.sum(gt, 1)

        return self.loss(h, gt)

def sim_clr_loss(h, h_corr, temperature = .5, distance_function = CosineSimilarityMatrix(), device = 'cuda', max_clr = 4.53):
    # TODO: Don't hardcode the max: Infer from batch size (log2(exp(0)/((BS-1) * exp(1))))
    distances = distance_function(h, h_corr) / temperature
    eyed = torch.eye(h.shape[0]).to(device)
    distances = torch.exp(distances)
    numerator = distances * eyed
    denominator = distances * (1 - eyed)

    denominator = torch.sum(denominator, dim = 1)
    numerator = torch.sum(numerator, dim = 1)
    res =  -torch.log(numerator / denominator)
    return (torch.sum(res) / h.shape[0]) / max_clr

def mse_rank_loss(h, target, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5, weighting = None, maxy = 3):

    sm = similarity(h, h)
    indicator = smooth_rank(sm, k, indicator_function)
    
    # Ground-truth Ranking function
    gt_sim = similarity(target, target)
    gt_indicator = smooth_rank(gt_sim, k_gt, indicator_function)
    error = torch.mean((indicator - gt_indicator) ** 2, dim = 1) / h.shape[0]
    return torch.mean(error, dim = 0) 
    

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


def rank_correlation_loss(h, target, indicator_function = sigmoid, similarity = CosineSimilarityMatrix(), scale = True, k = 1e-3, k_gt = 1e-5, weighting = None, maxy = 3):

    sm = similarity(h, h)
    indicator = smooth_rank(sm, k, indicator_function)
    
    # Ground-truth Ranking function
    gt_sim = similarity(target, target)
    gt_indicator = smooth_rank(gt_sim, k_gt, indicator_function)

    n = h.shape[0] if scale else 1
    C = corrcoef(gt_indicator, indicator)
    if weighting is not None:

        bins = (torch.arange(h.shape[0] - 1) + 1) 
        delta = bins.repeat(h.shape[0], 1) *  maxy / (h.shape[0] - 1)        
        if weighting == 'sigmoid':

            W = 1 / (1 + torch.exp(delta))
        
        Idiff = torch.abs(indicator - gt_indicator) * W
        scalator = Idiff.sum(1) / W.sum(1) + 1
    
    else:
        scalator = 1

    return 1 - (torch.sum(C * scalator/ n))

    
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
