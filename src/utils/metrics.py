import torch.nn as nn
import torch
from torch import Tensor
from typing import *
import numpy as np
from scipy import stats

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

def sigmoid(x, k=1.0):
    exponent = -x/k
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1./(1. + torch.exp(exponent))
    return y

def knn(distances, k = 1):

    # Returns the K-th NN mask
    # Default: 1st NN for "with a little help from my friend"

    shape = distances.shape
    n = shape[0]
    nns = distances.argsort()[:, k]

    mask = torch.zeros(shape, dtype=bool)

    mask[torch.arange(n), nns] = True

    return mask

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

def batched_spearman_rank(h_rank, gt_rank):

    data = [stats.spearmanr(h_rank[m].cpu(), gt_rank[m].cpu()) for m in range(h_rank.shape[0])]

    return [z.correlation for z in data], [z.pvalue for z in data]

def cov(m):
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.shape[-1] - 1)  # 1 / N
    meant = m - torch.mean(m, dim=(1, 2), keepdim=True)
    mt = torch.transpose(meant, 1, 2)  # if complex: mt = m.t().conj()
    return fact * meant.matmul(mt).squeeze()



def corrcoef(x, y, eps = 1e-4):
    # thanks https://discuss.pytorch.org/t/spearmans-correlation/91931/2
    batch_size = x.shape[0]
    x = torch.stack((x, y), 1)
    # calculate covariance matrix of rows
    c = cov(x)
    # normalize covariance matrix
    d = torch.diagonal(c, dim1=1, dim2=2)
    stddev = torch.pow(d, 0.5)
    stddev = stddev.repeat(1, 2).view(batch_size, 2, 2) + eps
    c = c.div(stddev)
    c = c.div(torch.transpose(stddev, 1, 2))
    return c[:, 1, 0]
