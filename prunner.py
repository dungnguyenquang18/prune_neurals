import torch.nn as nn
from .methods import base_method_coreset, kmean_prune, distance_based_clustering_prune, kmedoids
import torch

class Prunner():
    def __init__(self):
        self.method_name = {
            'Prune neurels': ['base', 'kmeans', 'distance-based-clustering, kmedoids'],
            'Prune dataset': ['base', 'kmeans', 'distance-based-clustering, kmedoids']
                            } 
    
    
    
    def prune_neurals(self, layer1:nn.Linear, layer2:nn.Linear,prune_ratio: float, method: str, device: None):
        try:    
            layer1.out_features != layer2.in_features
        except Exception as e:
            print(f"InOutFeatureError: the out feature of layer 1 is {layer1.out_features} but the in feature of layer 2 is {layer2.in_features}")
            
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

        coreset = None
        
        try:
            method in self.method_name
            if method == 'base':
                coreset = base_method_coreset
            elif method == 'kmeans':
                coreset = kmean_prune
            elif method == 'distance_based_clustering_prune':
                coreset = distance_based_clustering_prune
            else:
                coreset = kmedoids
        except Exception as e:
            print(f'Method name eror. The method name must be in {self.method_name}')
            
            

        
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


    def prune_dataset(self, matrix: torch.Tensor, prune_ratio: float, method: str, device: None):
        
            
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

        coreset = None
        
        try:
            method in self.method_name
            if method == 'base':
                coreset = base_method_coreset
            elif method == 'kmeans':
                coreset = kmean_prune
            elif method == 'distance_based_clustering_prune':
                coreset = distance_based_clustering_prune
            else:
                coreset = kmedoids
        except Exception as e:
            print(f'Method name eror. The method name must be in {self.method_name}')
        
        n, _ = matrix.shape
        m = int(n*(1 - prune_ratio))  # Số nơ-ron giữ lại    

        
        print(f"Starting pruning data prune_ratio={prune_ratio}...")
        
        _, _, select_indices = coreset(matrix, m)
        
        return select_indices

 
        