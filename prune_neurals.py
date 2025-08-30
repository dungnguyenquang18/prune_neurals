import torch.nn as nn
from .utils import base_method_coreset, kmean_prune
import torch

class PruneNeurals():
    def __init__(self):
        pass
    
    
    
    def prune(self, layer1:nn.Linear, layer2:nn.Linear,prune_ratio: float, method: str, device: None):
        try:    
            layer1.out_features != layer2.in_features
        except Exception as e:
            print(f"InOutFeatureError: the out feature of layer 1 is {layer1.out_features} but the in feature of layer 2 is {layer2.in_features}")
            
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")



        
        print(f"Starting pruning layer 1 with prune_ratio={prune_ratio}...")
        
        W = layer1.weight.data.cpu().numpy()  # Trọng số của tầng hiện tại
        n, d = W.shape
        m = int(n * (1 - prune_ratio))  # Số nơ-ron giữ lại
        coreset = None
        if method == 'base':
            coreset = base_method_coreset
        
        elif method == 'kmeans':
            coreset = kmean_prune
        else:
            
            pass
        
        C, u, sampled_indices = coreset(W, m)
        
        # Cập nhật tầng hiện tại (layer l)
        new_out_features = len(C)
        new_W = torch.zeros((new_out_features, d), device=device)
        new_bias = torch.zeros(new_out_features, device=device)
        
        # Nhân trọng số của tầng hiện tại với u để điều chỉnh đầu ra
        u_tensor = torch.tensor(u, device=device, dtype=torch.float32)
        for i, idx in enumerate(sampled_indices):
            new_W[i] = layer1.weight.data[idx] * u_tensor[i]  # Nhân với u tại tầng hiện tại
            new_bias[i] = layer1.bias.data[idx] * u_tensor[i]  # Nhân bias với u
        
        new_layer = nn.Linear(d, new_out_features)
        new_layer.weight.data = new_W
        new_layer.bias.data = new_bias
 
        
        # Tìm và cập nhật tầng Linear tiếp theo (layer l+1)
        

        # Chọn các cột của ma trận trọng số tương ứng với các nơ-ron được giữ lại
        selected_weights = layer2.weight.data[:, sampled_indices].to(device=device)
        # Nhân các cột với u
        new_next_W = selected_weights * u_tensor.unsqueeze(0)  # Broadcasting: (out_features, m) * (1, m)
        # Tạo tầng mới với kích thước phù hợp
        new_next_layer = nn.Linear(new_out_features, layer2.out_features)
        new_next_layer.weight.data = new_next_W
        new_next_layer.bias.data = layer2.bias.data  # Giữ nguyên bias

        
        
        # print(f"Pruning layer {layer_idx} completed!")
        return new_layer, new_next_layer


        