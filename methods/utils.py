import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing






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




def kmeans(points, k, max_iters=100, tol=1e-4, device=None):
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


def kmedoids(points, k, max_iters=100, device=None):
    if device is None:
        device = points.device
    else:
        points = points.to(device)

    n, d = points.shape
    indices = torch.randperm(n, device=device)[:k]
    medoids = points[indices]

    for _ in range(max_iters):
        # Gán điểm vào cụm gần nhất
        distances = torch.cdist(points, medoids)  # [n, k]
        labels = torch.argmin(distances, dim=1)   # [n]

        new_medoids = []
        for i in range(k):
            cluster_points = points[labels == i]
            if cluster_points.shape[0] == 0:  
                # Nếu cụm rỗng, giữ nguyên medoid cũ
                new_medoids.append(medoids[i])
                continue

            # Tính tổng khoảng cách giữa mỗi điểm và các điểm khác trong cụm
            intra_dist = torch.cdist(cluster_points, cluster_points)  # [m, m]
            total_dist = intra_dist.sum(dim=1)                        # [m]
            best_idx = torch.argmin(total_dist)
            new_medoids.append(cluster_points[best_idx])

        new_medoids = torch.stack(new_medoids)
        if torch.equal(new_medoids, medoids):
            break
        medoids = new_medoids

    # Gom thành k cụm (list các tensor)
    clusters = [points[labels == i] for i in range(k)]

    return clusters, medoids


def distance_based_clustering(points: torch.Tensor, k: int):
    """
    Cluster points into k groups based on their distance from the centroid.
    The points are sorted by their distance to the centroid and divided into k groups.

    Args:
        points (torch.Tensor): Tensor of shape (n, d) representing n points in d dimensions.
        k (int): Number of clusters.

    Returns:
        List[torch.Tensor]: List of k tensors, each containing the points in one cluster.
        torch.Tensor: Centroid of the dataset.
    """
    n, d = points.shape
    centroid = points.mean(dim=0)
    distances = torch.norm(points - centroid, dim=1)
    sorted_indices = torch.argsort(distances, descending=True)
    clusters = []
    group_size = n // k
    for i in range(k):
        start = i * group_size
        end = (i + 1) * group_size if i < k - 1 else n
        group_indices = sorted_indices[start:end]
        clusters.append(points[group_indices])
    return clusters, centroid




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



# Hàm tính MVEE (Khachiyan algorithm)
def compute_MVEE(P: torch.Tensor, tolerance=1e-3, max_iter=1000):
    """
    Tính Minimum Volume Enclosing Ellipsoid (MVEE) cho tập điểm P.
    
    Args:
        P (torch.Tensor): Tensor kích thước (N, d) chứa N điểm dữ liệu d-chiều.
        tolerance (float): Ngưỡng sai số để dừng thuật toán.
        max_iter (int): Số vòng lặp tối đa.
        
    Returns:
        G (torch.Tensor): Ma trận hình dạng (d, d) xác định ellipsoid (x-c)^T G (x-c) <= 1.
        c (torch.Tensor): Vector tâm ellipsoid (d,).
        vertices (torch.Tensor): Tensor chứa 2*d điểm giao của các bán trục với mặt ellipsoid.
    """
    # Đảm bảo dữ liệu là kiểu float
    if P.dtype != torch.float32 and P.dtype != torch.float64:
        P = P.float()
        
    N, d = P.shape
    
    # Handle edge cases
    if N <= 1:
        c = P.mean(dim=0) if N == 1 else torch.zeros(d, device=P.device, dtype=P.dtype)
        G = torch.eye(d, device=P.device, dtype=P.dtype)
        vertices = P.unsqueeze(0) if N == 1 else torch.zeros(1, d, device=P.device, dtype=P.dtype)
        return G, c, vertices
    
    # Check if all points are the same
    if torch.allclose(P, P[0].unsqueeze(0), atol=1e-8):
        c = P[0]
        G = torch.eye(d, device=P.device, dtype=P.dtype)
        vertices = c.unsqueeze(0)
        return G, c, vertices
    
    # 1. Lift points to d+1 dimensions (Khachiyan lifting scheme)
    # Q có kích thước (d+1, N)
    Q = torch.vstack([P.t(), torch.ones(1, N, device=P.device, dtype=P.dtype)])
    
    # 2. Khởi tạo trọng số u (uniform distribution)
    u = torch.ones(N, device=P.device, dtype=P.dtype) / N
    
    # Kích thước không gian nâng
    n_lifted = d + 1 
    
    # 3. Vòng lặp Khachiyan (Frank-Wolfe algorithm)
    for i in range(max_iter):
        # Tính ma trận hiệp phương sai có trọng số trong không gian nâng: X = Q diag(u) Q^T
        # Cách tính hiệu quả: X = (Q * u) @ Q.T
        X = (Q * u) @ Q.t()
        
        # Nghịch đảo ma trận X
        try:
            M_inv = torch.linalg.inv(X)
        except RuntimeError:
            # Xử lý trường hợp ma trận suy biến (thường do dữ liệu đồng phẳng hoặc quá ít điểm)
            # Thêm nhiễu nhỏ vào đường chéo
            X = X + torch.eye(n_lifted, device=P.device, dtype=P.dtype) * 1e-6
            M_inv = torch.linalg.inv(X)

        # Tính variance của mỗi điểm: V_i = q_i^T * M_inv * q_i
        # Đây là đường chéo của kết quả Q.T @ M_inv @ Q
        M_Q = M_inv @ Q
        variances = torch.sum(Q * M_Q, dim=0) # (N,)
        
        # Tìm điểm có variance lớn nhất (điểm xa nhất theo chuẩn Mahalanobis hiện tại)
        max_var_idx = torch.argmax(variances)
        max_var = variances[max_var_idx]
        
        # Điều kiện dừng: max_var <= (d+1)(1 + tolerance)
        # Theo lý thuyết, tại điểm tối ưu, max_var = d+1
        if max_var <= n_lifted * (1 + tolerance):
            break
            
        # Tính bước nhảy beta (step size)
        # Công thức cập nhật tối ưu cho thuật toán Khachiyan
        beta = (max_var - n_lifted) / ((n_lifted) * (max_var - 1))
        
        # Cập nhật trọng số u
        u = (1 - beta) * u
        u[max_var_idx] += beta

    # 4. Khôi phục tham số Ellipsoid từ trọng số u
    # Tâm c = P^T * u (trung bình có trọng số)
    c = P.t() @ u
    
    # Tính ma trận hiệp phương sai thực tế của dữ liệu gốc (d x d)
    # Sigma = (P - c)^T diag(u) (P - c)
    P_centered = P.t() - c.unsqueeze(1)
    Sigma = (P_centered * u) @ P_centered.t()
    
    # Đảm bảo Sigma là positive definite trước khi tính eigenvalues
    eigenvalues_sigma, eigenvectors_sigma = torch.linalg.eigh(Sigma)
    eigenvalues_sigma = torch.clamp(eigenvalues_sigma, min=1e-8)  # Ensure positive eigenvalues
    
    # Tái tạo Sigma với eigenvalues đã regularize
    Sigma = eigenvectors_sigma @ torch.diag(eigenvalues_sigma) @ eigenvectors_sigma.T
    
    # Ma trận hình dạng G = Sigma^-1 (không chia cho d, giống hàm cũ)
    # Phương trình Ellipsoid: (x-c)^T G (x-c) <= 1
    try:
        G = torch.linalg.inv(Sigma)
    except RuntimeError:
        # Nếu Sigma singular, thêm regularization
        Sigma = Sigma + torch.eye(d, device=P.device, dtype=P.dtype) * 1e-6
        G = torch.linalg.inv(Sigma)
    
    # 5. Tính các điểm "vertices" (giao điểm các bán trục)
    # Vertices được tính từ eigenvalues của Sigma (giống hàm cũ)
    # Radii = sqrt(eigenvalues của Sigma)
    sqrt_eigenvalues = torch.sqrt(eigenvalues_sigma)
    
    # Tính các đỉnh: c +/- sqrt(eigenvalue) * eigenvector
    verts = []
    for i in range(d):
        v_i = eigenvectors_sigma[:, i]  # Vector riêng thứ i của Sigma
        # Thêm hai điểm: +sqrt(λ_i) * v_i và -sqrt(λ_i) * v_i
        verts.append(c + sqrt_eigenvalues[i] * v_i)
        verts.append(c - sqrt_eigenvalues[i] * v_i)
        
    vertices = torch.stack(verts)  # Kích thước: (2*d, d)

    return G, c, vertices

def caratheodory_set(v, P, r):
    """
    Computes the Caratheodory set for point v in the convex hull of points P.
    :param v: torch.tensor or np.array of shape (d,), the target point.
    :param P: torch.tensor or np.array of shape (n, d), the set of points.
    :param r: int, the rank of P (dimension of the affine subspace).
    :return: torch.tensor of indices from P that form the Caratheodory set, or empty tensor if infeasible.
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
    c = np.zeros(n)  # Minimize sum of weights (feasibility problem)
    
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')    
    
    if not res.success:
        return torch.tensor([], dtype=torch.long)  # Return empty tensor if no feasible solution
    
    u = res.x
    support = np.where(u > tol)[0]
    
    if len(support) == 0:
        return torch.tensor([], dtype=torch.long)  # Return empty if no significant weights
    
    # Sparsify until size <= r + 1
    points = P[support]
    u = u[support]
    
    while len(support) > r + 1:
        m = len(support)
        A = np.vstack((points.T, np.ones(m)))
        
        # Find a vector in the null space using SVD
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
        if s[-1] > 1e-6:  # Check if rank is full (numerical tolerance)
            return torch.tensor([], dtype=torch.long)  # Infeasible if no null space
        
        alpha = Vt[-1, :]
        
        # Ensure there are positive components (flip if necessary)
        pos_idx = alpha > 0
        if not np.any(pos_idx):
            alpha = -alpha
            pos_idx = alpha > 0
            if not np.any(pos_idx):
                return torch.tensor([], dtype=torch.long)  # No valid direction
        
        # Compute t = min(u[i] / alpha[i] for alpha[i] > 0)
        t_candidates = u[pos_idx] / alpha[pos_idx]
        if len(t_candidates) == 0:
            return torch.tensor([], dtype=torch.long)  # No valid t
        t = np.min(t_candidates)
        
        # Update u = u - t * alpha
        u -= t * alpha
        u = np.maximum(u, 0)  # Clip negative due to numerical issues
        
        # Remove points with u <= tol
        keep = u > tol
        if not np.any(keep):
            return torch.tensor([], dtype=torch.long)  # No points remain
        u = u[keep]
        points = points[keep]
        support = support[keep]
    
    return torch.tensor(support, dtype=torch.long)

# Thuật toán l∞-CORESET
def l_infty_coreset(P):
    print(f"Running l∞-CORESET on matrix of shape {P.shape}...")
    
    # Đảm bảo P là tensor
    if not isinstance(P, torch.Tensor):
        raise TypeError(f"P must be a torch.Tensor, got {type(P)}")
    
    n, d = P.shape
    
    # Handle edge cases
    if n <= 1:
        return [0] if n == 1 else []
    
    r = torch.linalg.matrix_rank(P).item()
    r = min(r, d)  # Ensure r doesn't exceed dimensions

    # Bước 1: chiếu P về không gian affine bậc r
    if r < d:
        P_prime, _ = pca(P, r)
    else:
        P_prime = P

    # Bước 2: tính MVEE trong không gian r chiều
    G, c, vertices = compute_MVEE(P_prime)

    # Bước 3: tìm coreset
    S_prime = set()
    valid_vertices = 0
    vertices_with_coreset = 0
    for v in vertices:
        # Check if vertex contains invalid values
        if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
            continue
        valid_vertices += 1
            
        K = caratheodory_set(v, P_prime, r)
        if len(K) > 0:
            vertices_with_coreset += 1
        for x in K:
            S_prime.add(x.item() if hasattr(x, 'item') else x)

    print(f"l∞-CORESET: {valid_vertices}/{len(vertices)} valid vertices, {vertices_with_coreset} found coreset, selected {len(S_prime)} points.")
    return sorted(list(S_prime))


