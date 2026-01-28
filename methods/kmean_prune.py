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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

from .utils import caratheodory_set, compute_mvee_torch, compute_rank, l_infty_coreset, pca, kmeans




#step 2 - Cluster:
# Estimate the number of clusters(k): k = n // (d * 50)

def cluster(P):
    n, d = P.shape
    # Estimate number of clusters
    k = n // (d * 50)
    print(f"Estimated number of clusters (k): {k+1}")
    # Here you would implement your clustering algorithm, e.g., k-means
    # For simplicity, we will just return the estimated k
    return kmeans(P, k+1)





def process_cluster(cluster_idx, cluster_, mappingP, mappingQ, r, i):
    """
    Worker function để xử lý một cluster song song.
    Trả về list các tuple (orig_idxP, orig_idxQ, sensitivity_value)
    """
    results = []
    S = l_infty_coreset(cluster_)
    
    for idx in S:
        point = cluster_[idx]
        point_tuple = tuple(point.cpu().numpy().round(8))
        orig_idxP = mappingP[point_tuple]
        orig_idxQ = mappingQ[point_tuple]
        sensitivity_value = (2 * (r ** 1.5)) / i
        results.append((orig_idxP, orig_idxQ, sensitivity_value))
    
    return results


# Algorithm 2: CORESET
def kmean_prune(P, m, max_workers=None):
    """
    CORESET với hỗ trợ đa luồng.
    
    Args:
        P: Ma trận input
        m: Số điểm cần chọn
        max_workers: Số luồng tối đa (None = số CPU cores)
    """
    
    if max_workers is None:
        max_workers = 1
    
    print(f"Running CORESET to select {m} points from matrix of shape {P.shape}...")
    print(f"Using {max_workers} worker threads for parallel processing")
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
        
        # Xử lý song song các cluster
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tất cả các cluster tasks
            future_to_cluster = {
                executor.submit(process_cluster, cluster_idx, cluster_, mappingP, mappingQ, r, i): cluster_idx
                for cluster_idx, cluster_ in enumerate(clusters)
            }
            
            # Thu thập kết quả khi các task hoàn thành
            for future in as_completed(future_to_cluster):
                cluster_idx = future_to_cluster[future]
                try:
                    results = future.result()
                    # Cập nhật các giá trị từ kết quả
                    for orig_idxP, orig_idxQ, sensitivity_value in results:
                        usedP[orig_idxP] = True
                        usedQ[orig_idxQ] = True
                        s[orig_idxP] = sensitivity_value
                        
                        
                        
                except Exception as exc:
                    print(f"Cluster {cluster_idx} generated an exception: {exc}")
        
        # Kiểm tra trước khi tiếp tục vòng lặp
        if usedP.sum() >= m:
            break
        
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
    
    
