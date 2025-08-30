import torch.nn as nn
from .methods import base_method_coreset, kmean_prune
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

        coreset = None
        if method == 'base':
            coreset = base_method_coreset
        
        elif method == 'kmeans':
            coreset = kmean_prune
        else:
            
            pass

        
        print(f"Starting pruning layer 1 with prune_ratio={prune_ratio}...")
        
        W = layer2.weight.data.T.cpu()  # shape: l2, l3
        print(f"W shape: {W.shape}")
        l2, l3 = W.shape #need to modify d
        m = int(l2 * (1 - prune_ratio))  # Số nơ-ron giữ lại
        
        C, u, sampled_indices = coreset(W, m) #shape: m, l3
        u_tensor = torch.tensor(u, device=device, dtype=torch.float32)
        new_l2 = len(C)
        new_W = W[sampled_indices].to(device) #shape: new_l2, l3
        # Apply weights along the first dimension (new_l2)
        new_W = new_W * u_tensor.unsqueeze(1)  # u_tensor: (new_l2,) -> (new_l2, 1) for broadcasting
        new_W = new_W.T #shape: l3, new_l2
        new_layer2 = nn.Linear(new_l2, l3)
        new_layer2.weight.data = new_W
        new_layer2.bias.data = layer2.bias.data
        
        
        new_layer1 = nn.Linear(layer1.in_features, new_l2)
        new_layer1.weight.data = layer1.weight.data[sampled_indices]
        new_layer1.bias.data = layer1.bias.data[sampled_indices]
        
        
        
        
        

        
        
        # print(f"Pruning layer {layer_idx} completed!")
        return new_layer1, new_layer2


        