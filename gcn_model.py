import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.3):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(adj, x)     # Neighborhood aggregation
        x = self.linear(x)           # Transformation
        return F.relu(x)             # Activation


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, dropout)

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        return x






