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
    distances = torch.norm(P - v, dim=1)
    idx = torch.topk(distances, r + 1, largest=False).indices
    return P[idx]

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
def base_method_coreset(P, m):
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


