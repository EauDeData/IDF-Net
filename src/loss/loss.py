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
    pass

class PearsonLoss(CustomLoss):
    pass

class OrthAligment(CustomLoss):
    def __init__(self, p = 2) -> None:
        self.p = p

    def forward(self, x, gt):
        return orth_aligment(x, gt, self.p)

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

def inequality_satisfied_loss():

    #############################################
    # Demonstrated in whiteboard (i think)      #
    # Nn and Nm are mutually close.             #
    # Nx is not.                                #
    # Loss defined by (Nm, Nn, Nx) triplets     #
    # S is a similarity function.               #
    # E = S(Nn, Nx)                             #
    #                                           #
    # Loss = -2E + [S(Nm, Nx) + S(Nn, Nx)]      #
    #                                           #
    # Loss will be 0 if E is 0                  #
    # and Nm, Nn are 0-similar to Nx            #
    #                                           #
    # Question:                                 #
    #    In some situations Nx is mutual to Nm  #
    # Because it only takes into consideration  #  
    # Nodes in the triplet. How do we scale to  #
    # A continuous scenario?                    #
    #############################################

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