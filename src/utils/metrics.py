import torch.nn as nn
import torch
from torch import Tensor
from typing import *
import numpy as np

class CosineSimilarityMatrix(nn.Module):
    name = 'cosine_matrix'
    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super(CosineSimilarityMatrix, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return cosine_similarity_matrix(x1, x2, self.dim, self.eps)

class EuclideanSimilarityMatrix(nn.Module):
    name = 'euclidean_matrix'
    def __init__(self, dim: int = 1, eps: float = 1) -> None:
        super(EuclideanSimilarityMatrix, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return euclidean_similarity_matrix(x1, x2, self.eps)

class EuclideanDistanceMatrix(nn.Module):
    name = 'euclidean_distance_matrix'
    def __init__(self,) -> None:
        super(EuclideanDistanceMatrix, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return euclidean_distance_matrix(x1, x2,)

def cosine_similarity_matrix(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    '''
    When using cosine similarity the constant value must be positive
    '''
    #Cosine sim:
    xn1, xn2 = torch.norm(x1, dim=dim), torch.norm(x2, dim=dim)
    x1 = x1 / torch.clamp(xn1, min=eps).unsqueeze(dim)
    x2 = x2 / torch.clamp(xn2, min=eps).unsqueeze(dim)
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(1)

    sim = torch.tensordot(x1, x2, dims=([2], [2])).squeeze()

    sim = (sim + 1)/2 #range: [-1, 1] -> [0, 2] -> [0, 1]

    return sim

def euclidean_similarity_matrix(x1, x2, eps):
    return 1/(1+torch.cdist(x1, x2)+eps)

def euclidean_distance_matrix(x1, x2):
    return torch.cdist(x1, x2)

def knn(distances, k = 1):

    # Returns the K-th NN mask
    # Default: 1st NN for "with a little help from my friend"

    shape = distances.shape
    n = shape[0]
    nns = distances.argsort()[:, k]

    mask = torch.zeros(shape, dtype=bool)

    mask[torch.arange(n), nns] = True

    return mask * 1

def mutual_knn(distances, k):

    # Define the value of k
    # TODO: I think this is not ok
    k = 3
    n = distances.shape[0]

    # Find the indices of the k nearest neighbors for each point
    nearest_neighbors = np.argsort(distances, axis=1)[:, :k]

    # Create a boolean mask indicating whether each pair of points is mutual nearest neighbors
    mask = np.zeros(distances.shape, dtype=bool)
    mask[np.arange(n)[:, np.newaxis], nearest_neighbors] = True
    mask[nearest_neighbors, np.arange(n)[:, np.newaxis]] = True

    # Use the boolean mask to create the mutual KNN adjacency matrix
    adjacency_matrix = mask.astype(int)
    np.fill_diagonal(adjacency_matrix, 0)

    return adjacency_matrix
