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
    
    # Đảm bảo P là tensor và có device
    if not isinstance(P, torch.Tensor):
        raise TypeError(f"P must be a torch.Tensor, got {type(P)}")
    
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


def kmeans_torch(points, k, max_iters=100, tol=1e-4, device=None):
    if device is None:
        device = points.device
    else:
        points = points.to(device)

    n, d = points.shape
    indices = torch.randperm(n, device=device)[:k]
    centroids = points[indices]

    for _ in range(max_iters):
        distances = torch.cdist(points, centroids)  # [n, k]
        labels = torch.argmin(distances, dim=1)     # [n]

        new_centroids = torch.stack([
            points[labels == i].mean(dim=0) if torch.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        shift = torch.norm(new_centroids - centroids, dim=1).sum()
        centroids = new_centroids
        if shift < tol:
            break

    # Gom thành k cụm (list các tensor)
    clusters = [points[labels == i] for i in range(k)]

    return clusters, centroids



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
    """
    Computes the Caratheodory set for point v in the convex hull of points P.
    :param v: torch.tensor or np.array of shape (d,), the target point.
    :param P: torch.tensor or np.array of shape (n, d), the set of points.
    :param r: int, the rank of P (dimension of the affine subspace).
    :return: torch.tensor of indices from P that form the Caratheodory set.
    """
    # Convert to numpy if torch tensors
    if hasattr(P, 'numpy'):
        P = P.cpu().numpy()
    else:
        P = np.array(P)
    
    if hasattr(v, 'numpy'):
        v = v.cpu().numpy()
    else:
        v = np.array(v)
    
    n, d = P.shape
    tol = 1e-8
    
    # Set up the linear program to find initial weights u >= 0 such that P.T @ u = v and sum(u) = 1
    A_eq = np.vstack((P.T, np.ones(n)))
    b_eq = np.hstack((v, 1))
    c = np.zeros(n)  # No objective, just feasibility
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    
    if not res.success:
        raise ValueError("No feasible solution found for the convex combination.")
    
    u = res.x
    support = np.where(u > tol)[0]
    u = u[support]
    points = P[support]
    
    # Sparsify until size <= r + 1
    while len(support) > r + 1:
        m = len(support)
        
        # Construct matrix A = [points.T; ones(1, m)]
        A = np.vstack((points.T, np.ones(m)))
        
        # Find a vector in the null space using SVD
        _, s, Vt = svd(A, full_matrices=False)
        if s[-1] > 1e-6:
            raise ValueError("Unexpected full rank in null space computation.")
        
        alpha = Vt[-1, :]
        
        # Ensure there are positive components (flip if necessary)
        pos_idx = alpha > 0
        if not np.any(pos_idx):
            alpha = -alpha
            pos_idx = alpha > 0
            if not np.any(pos_idx):
                raise ValueError("No positive direction in alpha.")
        
        # Compute t = min(u[i] / alpha[i] for alpha[i] > 0)
        t_candidates = u[pos_idx] / alpha[pos_idx]
        t = np.min(t_candidates)
        
        # Update u = u - t * alpha
        u -= t * alpha
        u = np.maximum(u, 0)  # Clip negative due to numerical issues
        
        # Remove points with u <= tol
        keep = u > tol
        u = u[keep]
        points = points[keep]
        support = support[keep]
    
    return torch.tensor(support)

# Thuật toán l∞-CORESET
def l_infty_coreset(P):
    print(f"Running l∞-CORESET on matrix of shape {P.shape}...")
    
    # Đảm bảo P là tensor
    if not isinstance(P, torch.Tensor):
        raise TypeError(f"P must be a torch.Tensor, got {type(P)}")
    
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
def kmean_prune(P, m):
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
    
    
