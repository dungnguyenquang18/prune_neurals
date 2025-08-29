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



# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hàm tải mô hình VGG16 đã huấn luyện trước
def load_pretrained_vgg16(model_path):
    print("Starting to load pretrained VGG16 model...")
    model = torchvision.models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 10)
    state_dict = torch.load(model_path, weights_only=True)
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully!")
    return model.to(device)

#step 1 - PCA: If the dimension of P is too big -> Using PCA to reduce dimention of P
def pca(P: torch.Tensor, new_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reduce dimension of tensor P using sklearn PCA.
    Returns reduced data (torch.Tensor) and explained variances (torch.Tensor).
    """
    print(f"starting reduce dimensions from {P.shape[1]} to {new_dim}")
    device = P.device
    
    # Chuyển sang numpy
    P_np = P.detach().cpu().numpy()
    
    # Fit PCA
    new_dim = min(new_dim, P.shape[1])
    pca_model = PCA(n_components=new_dim)
    P_reduced_np = pca_model.fit_transform(P_np)
    
    # explained_variance_ tương đương với giá trị riêng (eigenvalues)
    explained_var = pca_model.explained_variance_
    
    # Chuyển kết quả về torch.Tensor
    P_reduced = torch.tensor(P_reduced_np, dtype=torch.float32, device=device)
    explained_var_torch = torch.tensor(explained_var, dtype=torch.float32, device=device)
    
    return P_reduced, explained_var_torch


#step 2 - Cluster:
# Estimate the number of clusters(k): k = n // (d * 50)

def cluster(P):
    n, d = P.shape
    # Estimate number of clusters
    k = n // (d * 50)
    print(f"Estimated number of clusters (k): {k+1}")
    # Here you would implement your clustering algorithm, e.g., k-means
    # For simplicity, we will just return the estimated k
    return kmeans_torch(P, k+1)


def kmeans_torch(P, k, max_iters=100, tol=1e-4, device=None):
    mu = torch.mean(P, dim = 0)
    diff = P - mu
    dist = torch.norm(diff, dim=1)
    _, indices = torch.sort(dist)
    clusters = []
    # print(indices.shape)
    n = P.shape[0]
    l = n//k
    # print(l)
    for i in range(0, k):
        # print(f"{i*k}  {i*l+l}")
        clusters.append(P[indices[i*l:i*l+l]])
    
    if n%k != 0:
        clusters.append(P[indices[k*l:]])
    
    
    return clusters, _


def compute_rank(matrix, device=None):
    """
    Compute the rank of a matrix.

    Parameters:
    matrix (torch.Tensor or numpy array): Input matrix
    device: Device to perform computation on

    Returns:
    int: Rank of the matrix
    """
    if isinstance(matrix, torch.Tensor):
        matrix_cpu = matrix.cpu().numpy()
    else:
        matrix_cpu = matrix
    
    return np.linalg.matrix_rank(matrix_cpu)

# Hàm tính MVEE
def compute_mvee_torch(P_prime):
    n, d = P_prime.shape
    P_np = P_prime.cpu().numpy()

    # Sử dụng EllipticEnvelope để tính Löwner ellipsoid
    clf = EllipticEnvelope(support_fraction=1.0, contamination=0.01, random_state=42)
    clf.fit(P_np)
    c_np = clf.location_
    Sigma_np = clf.covariance_

    c = torch.from_numpy(c_np).float()
    Sigma = torch.from_numpy(Sigma_np).float()
    G = torch.linalg.inv(Sigma)

    # Tính 2*d vertices: các điểm tiếp xúc của ellipsoid với các mặt phẳng theo trục chính
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    vertices_list = []
    for i in range(d):
        v_i = eigenvectors[:, i]
        # Thêm hai điểm: +sqrt(λ_i) * v_i và -sqrt(λ_i) * v_i
        vertices_list.append(c + sqrt_eigenvalues[i] * v_i)
        vertices_list.append(c - sqrt_eigenvalues[i] * v_i)
    vertices = torch.stack(vertices_list)  # Kích thước: (2*d, d)

    return G, c, vertices

# Hàm tìm Carathéodory set
def caratheodory_set(v, P, r):
    distances = torch.norm(P - v, dim=1)
    idx = torch.topk(distances, r + 1, largest=False).indices
    return idx

# Thuật toán l∞-CORESET
def l_infty_coreset(P):
    print(f"Running l∞-CORESET on matrix of shape {P.shape}...")
    n, d = P.shape
    r = torch.linalg.matrix_rank(P).item()

    # Bước 1: chiếu P về không gian affine bậc r
    P_prime, _ = pca(P, 8)

    # Bước 2: tính MVEE trong không gian r chiều
    G, c, vertices = compute_mvee_torch(P_prime)

    # Bước 3: tìm coreset
    S_prime = set()
    for v in vertices:
        K = caratheodory_set(v, P_prime, r)
        for x in K:
            S_prime.add(x)
    


    

    print(f"l∞-CORESET completed, selected {len(S_prime)} points.")
    return sorted(list(S_prime))

# Algorithm 2: CORESET
def coreset(P, m):
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
        for cluster_ in clusters:
            S = l_infty_coreset(cluster_)

            # print(S)
            for idx in S:
                point = cluster_[idx]
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
    
    
