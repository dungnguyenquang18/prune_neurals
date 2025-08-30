import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import EllipticEnvelope

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

# Hàm chuyển đổi P → P'
def project_to_affine_subspace(P, r=None):
    """
    Chiếu P về không gian con affine bậc r
    Trả về:
        P_prime: (n, r)
        Y: (d, r)
        z: (d,)
    """
    z = P.mean(dim=0)  # gốc affine
    X = P - z          # dịch gốc
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)

    if r is None:
        r = torch.linalg.matrix_rank(X).item()

    Y = Vh[:r].T       # (d, r)
    P_prime = X @ Y    # (n, r)
    return P_prime, Y, z

def recover_from_projection(P_prime_subset, Y, z):
    """
    Phục hồi các điểm từ không gian chiếu P' về không gian gốc P
    Input:
        P_prime_subset: (m, r)
        Y: (d, r)
        z: (d,)
    Output:
        P_subset_original: (m, d)
    """
    return P_prime_subset @ Y.T + z

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
    n, d = P.shape
    r = torch.linalg.matrix_rank(P).item()

    # Bước 1: chiếu P về không gian affine bậc r
    P_prime, Y, z = project_to_affine_subspace(P, r)

    # Bước 2: tính MVEE trong không gian r chiều
    G, c, vertices = compute_mvee_torch(P_prime)

    # Bước 3: tìm coreset
    S_prime = set()
    for v in vertices:
        K = caratheodory_set(v, P_prime, r)
        for p in K:
            S_prime.add(tuple(p.tolist()))

    # Bước 4: ánh xạ lại về không gian gốc
    S_prime_tensor = torch.tensor(list(S_prime), device=P.device)
    S = recover_from_projection(S_prime_tensor, Y, z)

    print(f"l∞-CORESET completed, selected {len(S)} points.")
    return S

# Algorithm 2: CORESET
def base_method_coreset(P_, m):
    P = P_.numpy()
    print(f"Running CORESET to select {m} points from matrix of shape {P.shape}...")
    n, d = P.shape
    Q = P.copy()
    i = 1
    sensitivities = np.zeros(n)
    indices = np.arange(n)
    while len(Q) >= 4 * np.linalg.matrix_rank(Q)**2:
        S = l_infty_coreset(Q)
        r = np.linalg.matrix_rank(Q)
        S_indices = []
        for s in S:
            idx = np.where((Q == s).all(axis=1))[0][0]
            S_indices.append(idx)
            sensitivities[indices[idx]] = r**1.5 / i
        mask = np.ones(len(Q), dtype=bool)
        mask[S_indices] = False
        Q = Q[mask]
        indices = indices[mask]
        i += 1
    r = np.linalg.matrix_rank(Q)
    for j in range(len(Q)):
        sensitivities[indices[j]] = 2 * r**1.5 / i
    t = sensitivities.sum()
    probs = sensitivities / t
    sampled_indices = np.random.choice(n, size=m, replace=False, p=probs)
    C = P[sampled_indices]
    u = t / (m * probs[sampled_indices])
    print(f"CORESET completed, selected {len(C)} points.")
    return C, u, sampled_indices


