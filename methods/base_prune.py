import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.covariance import EllipticEnvelope
from scipy.optimize import linprog

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



def caratheodory_set(v, P, r):
    """
    Computes the Caratheodory set for point v in the convex hull of points P.
    :param v: torch.tensor or np.array of shape (d,), the target point.
    :param P: torch.tensor or np.array of shape (n, d), the set of points.
    :param r: int, the rank of P (dimension of the affine subspace).
    :return: torch.tensor of indices from P that form the Caratheodory set, or empty tensor if infeasible.
    """
    # Convert to numpy if torch tensors
    if hasattr(P, 'detach'):
        P = P.detach().cpu().numpy()
    else:
        P = np.array(P)

    if hasattr(v, 'detach'):
        v = v.detach().cpu().numpy()
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
    
    return np.array(support, dtype=np.int64)

# Thuật toán l∞-CORESET
def l_infty_coreset(P_input):
    print(f"Running l∞-CORESET on matrix of shape {P_input.shape}...")
    # Normalize input to torch tensor (on CPU for linear algebra and sklearn interop)
    if hasattr(P_input, 'detach'):
        P_torch = P_input.detach().cpu()
    else:
        P_torch = torch.from_numpy(np.array(P_input, dtype=np.float32))

    n, d = P_torch.shape
    r = torch.linalg.matrix_rank(P_torch).item()

    # Step 1: project to affine subspace of rank r
    P_prime, Y, z = project_to_affine_subspace(P_torch, r)

    # Step 2: compute MVEE in r-dimensional space
    G, c, vertices = compute_mvee_torch(P_prime)

    # Step 3: find coreset (collect indices in projected space)
    indices_set = set()
    for v in vertices:
        K = caratheodory_set(v, P_prime, r)  # indices into P_prime
        if K is None or len(K) == 0:
            continue
        for idx in np.atleast_1d(K):
            indices_set.add(int(idx))

    indices_array = np.array(sorted(indices_set), dtype=np.int64)
    print(f"l∞-CORESET completed, selected {len(indices_array)} points.")
    return indices_array

# Algorithm 2: CORESET
def base_method_coreset(P_, m):
    # Accept torch tensor or numpy array
    if hasattr(P_, 'detach'):
        P = P_.detach().cpu().numpy()
    else:
        P = np.array(P_, copy=True)
    print(f"Running CORESET to select {m} points from matrix of shape {P.shape}...")
    n, d = P.shape
    Q = P.copy()
    i = 1
    sensitivities = np.zeros(n)
    indices = np.arange(n)
    while len(Q) >= 4 * np.linalg.matrix_rank(Q)**2:
        S = l_infty_coreset(Q)  # indices into current Q
        r = np.linalg.matrix_rank(Q)
        S_indices = list(map(int, S))
        for idx in S_indices:
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


