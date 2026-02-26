import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import EllipticEnvelope
from scipy.optimize import linprog
from sklearn.decomposition import PCA
from typing import Tuple

from .utils import  compute_rank, l_infty_coreset, pca

# Algorithm 2: CORESET
def base_method_coreset(P_, m):
    """
    CORESET algorithm.
    
    Args:
        P_: Ma trận input (torch.Tensor hoặc numpy array)
        m: Số điểm cần chọn
    """
    # Accept torch tensor or numpy array
    if hasattr(P_, 'detach'):
        P = P_.detach().cpu()
    else:
        P = torch.from_numpy(np.array(P_, dtype=np.float32))
    
    print(f"Running CORESET to select {m} points from matrix of shape {P.shape}...")
    
    # Step 1: Giảm chiều bằng PCA nếu chiều > 8 (giống các thuật toán khác)
    if P.shape[1] > 8:
        Q, _ = pca(P, 8)
    else:
        Q = P
    Q = Q.cpu()
    
    n = P.shape[0]
    s = torch.zeros(n, dtype=torch.float32, device='cpu')
    
    # Tạo mapping từ điểm trong Q về index trong P
    mappingP = {}
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
        for j in range(l):
            mappingQ[tuple(Q[j].cpu().numpy().round(8))] = j
        
        # Gọi l_infty_coreset trên Q (đã được giảm chiều)
        S = l_infty_coreset(Q)
        
        for idx in S:
            point = Q[idx]
            point_tuple = tuple(point.cpu().numpy().round(8))
            orig_idxP = mappingP[point_tuple]
            orig_idxQ = mappingQ[point_tuple]
            
            usedP[orig_idxP] = True
            usedQ[orig_idxQ] = True
            s[orig_idxP] = (2 * (r ** 1.5)) / i
            
            
        
        # Update remaining points
        Q = Q[~usedQ]
        i += 1
        l = Q.shape[0]
        if l == 0:
            break
        r = compute_rank(Q)
        condition = 2 * (r ** 2)
    
    # Gán sensitivity cho các điểm còn lại
    if Q.shape[0] > 0:
        for j in range(Q.shape[0]):
            point_tuple = tuple(Q[j].cpu().numpy().round(8))
            if point_tuple in mappingP:
                s[mappingP[point_tuple]] = (2 * (r ** 1.5)) / i
    
    # Tính xác suất và lấy mẫu
    t = s.sum()
    probs = s / t
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()  # Đảm bảo tổng = 1
    
    sampled_indices = np.random.choice(n, size=m, replace=False, p=probs)
    C = P[sampled_indices]
    u = t / (m * probs[sampled_indices])
    
    print(f"CORESET completed, selected {len(C)} points.")
    return C, u, sampled_indices


