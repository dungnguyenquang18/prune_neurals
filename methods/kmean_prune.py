import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import EllipticEnvelope
from scipy.optimize import linprog
from typing import Tuple, List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .utils import  compute_rank, l_infty_coreset, pca, kmeans




#step 2 - Cluster:
# Estimate the number of clusters(k): k = n // (d * 50)

def cluster(P):
    n, d = P.shape
    # Estimate number of clusters
    # k = n // (d * 50)
    k = int(np.log2(n))
    print(f"Estimated number of clusters (k): {k+1} / {n} points")
    # Here you would implement your clustering algorithm, e.g., k-means
    # For simplicity, we will just return the estimated k
    return kmeans(P, k+1)





# Algorithm 2: CORESET
def kmean_prune(P, m):
    """
    CORESET algorithm.
    
    Args:
        P: Ma trận input
        m: Số điểm cần chọn
    """
    print(f"Running CORESET to select {m} points from matrix of shape {P.shape}...")
    if P.shape[0] > 8:
        Q, _ = pca(P, 8)
    else:
        Q = P
    Q = Q.cpu()
    n = P.shape[0]
    s = torch.zeros(n, dtype=torch.float32, device='cpu')
    mappingP = {}
    # print(Q[1])
    for i in range(n):
        mappingP[tuple(Q[i].cpu().numpy().round(8))] = i

    usedP = torch.zeros(n, dtype=torch.bool, device='cpu')
    
    i = 1
    l = Q.shape[0]
    r = compute_rank(Q)
    condition = 2 * (r ** 2)
    
    while l >= condition:
        usedQ = torch.zeros(l, dtype=torch.bool, device='cpu')
        mappingQ = {}
        # print(Q[1])
        for j in range(l):
            mappingQ[tuple(Q[j].cpu().numpy().round(8))] = j
        clusters, _ = cluster(Q)
        
        for cluster_idx, cluster_ in enumerate(clusters):
            S = l_infty_coreset(cluster_)
            for idx in S:
                point = cluster_[idx]
                point_tuple = tuple(point.cpu().numpy().round(8))
                orig_idxP = mappingP[point_tuple]
                orig_idxQ = mappingQ[point_tuple]
                sensitivity_value = (2 * (r ** 1.5)) / i
                usedP[orig_idxP] = True
                usedQ[orig_idxQ] = True
                s[orig_idxP] = sensitivity_value
        
        # Update remaining points
            
        Q = Q[~usedQ]
        i += 1
        l = Q.shape[0]
        r = compute_rank(Q)
        condition = 2 * (r ** 2)
        print(l)
    if Q.shape[0] > 0:
        for j in range(Q.shape[0]):
            s[mappingP[tuple(Q[j].cpu().numpy().round(8))]] = (2 * (r ** 1.5)) / i
    
    t = s.sum()
    probs = s / t
    probs = np.array(probs, dtype=np.float64)  # đảm bảo double precision
    probs = probs / probs.sum()

    sampled_indices = np.random.choice(n, size=m, replace=False, p=probs)
    C = P[sampled_indices]
    u = t / (m * probs[sampled_indices])
    print(f"CORESET completed, selected {len(C)} points.")
    return C, u, sampled_indices
    
    
