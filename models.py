import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool

class GATModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3, heads=4, dropout=0.1):
        super(GATModel, self).__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        
        # Initial node projection
        self.node_lin = nn.Linear(node_in_dim, hidden_dim)
        
        # Graph Transformer Layers
        # TransformerConv supports edge features with 'edge_dim' argument
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=edge_in_dim,
                dropout=dropout
            )
            self.layers.append(conv)
            
        self.dropout_layer = nn.Dropout(dropout)
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # x: [num_nodes, node_in_dim]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, edge_in_dim]
        
        x = self.node_lin(x)
        x = F.relu(x)
        
        for conv in self.layers:
            # TransformerConv output is [num_nodes, heads * out_channels] = [num_nodes, hidden_dim]
            x_new = conv(x, edge_index, edge_attr)
            x_new = F.relu(x_new)
            x_new = self.dropout_layer(x_new)
            x = x + x_new # Residual connection (optional, but good for deep GNNs)
            
        # Global Pooling
        x_pool = global_mean_pool(x, batch) # [batch_size, hidden_dim]
        
        # Prediction
        out = self.regressor(x_pool)
        return out.squeeze()


class SmilesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, max_len=128, dropout=0.1):
        super(SmilesTransformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, src):
        # src: [batch_size, seq_len]
        
        x = self.embedding(src) # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Pass through transformer
        # Masking for padding could be added, but for simplicity we rely on the model learning padding tokens are uniform
        x = self.transformer_encoder(x)
        
        # Global Pooling (Average over sequence, or take CLS token if we had one. Simple Avg here.)
        # Assuming padding tokens might affect avg, but let's effectively mean pool.
        x_pool = x.mean(dim=1) 
        
        out = self.regressor(x_pool)
        return out.squeeze()
