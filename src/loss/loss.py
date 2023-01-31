from src.utils.metrics import CosineSimilarityMatrix, EuclideanSimilarityMatrix, EuclideanDistanceMatrix
from src.utils.metrics import mutual_knn, knn

import torch
import torch.nn as nn
from typing import *
from torch.linalg import norm


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

def nns_loss(h, gt, distance_function = EuclideanDistanceMatrix(), temperature = 0.75):
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