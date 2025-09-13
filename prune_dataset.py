import torch.nn as nn
from .methods import base_method_coreset, kmean_prune, distance_based_clustering_prune, kmedoids
import torch

class PruneDataset():
    def __init__(self):
        self.method_name = ['base', 'kmeans', 'distance-based-clustering, kmedoids']
    
    
    
    def prune(self, matrix: torch.Tensor, prune_ratio: float, method: str, device: None):
        
            
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

 
        