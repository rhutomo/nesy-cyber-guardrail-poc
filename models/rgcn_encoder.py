import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv

class ThreatRGCN(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) for detecting 
    anomalous execution paths in heterogeneous system graphs.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_relations: int):
        super(ThreatRGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        # Layer 1: Relation-specific message passing
        x = self.conv1(x, edge_index, edge_type)
        x = self.relu(x)
        
        # Layer 2: Final embedding / risk score projection
        x = self.conv2(x, edge_index, edge_type)
        return self.sigmoid(x)