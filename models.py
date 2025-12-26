import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_add_pool

# class GATModel(nn.Module):
#     def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3, heads=4, dropout=0.1):
#         super(GATModel, self).__init__()
#         self.node_in_dim = node_in_dim
#         self.edge_in_dim = edge_in_dim
#         self.hidden_dim = hidden_dim
        
#         # Initial node projection
#         self.node_lin = nn.Linear(node_in_dim, hidden_dim)
        
#         # Graph Transformer Layers
#         # TransformerConv supports edge features with 'edge_dim' argument
#         self.layers = nn.ModuleList()
#         for _ in range(num_layers):
#             conv = TransformerConv(
#                 in_channels=hidden_dim, 
#                 out_channels=hidden_dim // heads, 
#                 heads=heads, 
#                 edge_dim=edge_in_dim,
#                 dropout=dropout
#             )
#             self.layers.append(conv)
            
#         self.dropout_layer = nn.Dropout(dropout)
        
#         # Regression Head
#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def forward(self, x, edge_index, edge_attr, batch):
#         # x: [num_nodes, node_in_dim]
#         # edge_index: [2, num_edges]
#         # edge_attr: [num_edges, edge_in_dim]
        
#         x = self.node_lin(x)
#         x = F.relu(x)
        
#         for conv in self.layers:
#             # TransformerConv output is [num_nodes, heads * out_channels] = [num_nodes, hidden_dim]
#             x_new = conv(x, edge_index, edge_attr)
#             x_new = F.relu(x_new)
#             x_new = self.dropout_layer(x_new)
#             x = x + x_new # Residual connection (optional, but good for deep GNNs)
            
#         # Global Pooling
#         x_pool = global_mean_pool(x, batch) # [batch_size, hidden_dim]
#         #using gem pooling with learnable temperature
#         #x_pool = gem_pooling(x_pool, batch, learnable_temp=True)
        
#         # Prediction
#         out = self.regressor(x_pool)
#         return out.squeeze()

class GATModel(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=64, num_layers=3, heads=4, dropout=0.1):
        super(GATModel, self).__init__()
        
        # 1. Node Embedding (Başlangıç Projeksiyonu)
        self.node_lin = nn.Linear(node_in_dim, hidden_dim)
        
        # 2. Graph Transformer Katmanları
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = TransformerConv(
                in_channels=hidden_dim, 
                out_channels=hidden_dim // heads, 
                heads=heads, 
                edge_dim=edge_in_dim, # Kenar özelliklerini burada kullanıyor
                dropout=dropout
            )
            self.layers.append(conv)
            # Batch Normalization (Eğitimi hızlandırır ve stabilize eder)
            self.norms.append(nn.BatchNorm1d(hidden_dim))
            
        # 3. Regressor Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(), # Modern aktivasyon (ReLU yerine)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1) # Çıktı: Erime Derecesi
        )

    def forward(self, data):
        # PyG Data objesinden parçaları al
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Başlangıç projeksiyonu
        x = self.node_lin(x)
        
        # Katmanlar döngüsü
        for conv, norm in zip(self.layers, self.norms):
            # Skip Connection (Residual) için x'i sakla
            x_in = x
            
            # Konvolüsyon
            x = conv(x, edge_index, edge_attr)
            
            # Normalization + Activation
            x = norm(x)
            x = torch.nn.functional.gelu(x) # Aktivasyon EKLENDİ!
            
            # Residual Connection (x = x + f(x))
            # Modelin derinleştikçe unutmasını engeller
            x = x + x_in 
            
        # --- KRİTİK KISIM: POOLING ---
        # Atom vektörlerinden -> Molekül vektörüne geçiş
        # Hem ortalamayı (mean) hem toplamı (add) kullanmak genelde daha iyidir ama
        # şimdilik sadece mean kullanalım:
        x_mean = global_mean_pool(x, batch)
        
        # Toplam özellikler (Kütle, boyut, toplam bağ sayısı)
        x_add = global_add_pool(x, batch)
        
        # İkisini birleştir (Concatenate)
        # Yan yana yapıştırıyoruz: [Mean Vektörü | Add Vektörü]
        x = torch.cat([x_mean, x_add], dim=1)
        
        # Regresyon
        return self.regressor(x)

class SmilesTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=3, dim_feedforward=512, max_len=128, dropout=0.1, activation='gelu', pad_idx=0):
        # ##EMİN## should try GELU and SwiGLU as activation function
        super(SmilesTransformer, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional Encoding
        # ##EMİN## should try fixed positional encoding RoPe or simple sin and cos
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len+1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Regression Head
        self.regressor = nn.Sequential(
            # ##EMİN## gemini were doing nn.Linear(d_model, d_model // 2)
            #  but i think it is better to do nn.Linear(d_model, d_model * 2) no it isnt
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
            # ##EMİN## since the melting point is always positive maybe we can use relu or softplus
            #nn.Softplus()
            #nn.ReLU()
        )

    # def forward(self, src):
    #     # src: [batch_size, seq_len]
        
    #     x = self.embedding(src) # [batch_size, seq_len, d_model]
        
    #     # Add positional encoding
    #     seq_len = x.size(1)
    #     x = x + self.pos_encoder[:, :seq_len, :]
        
    #     # Pass through transformer
    #     # Masking for padding could be added, but for simplicity we rely on the model learning padding tokens are uniform
    #     x = self.transformer_encoder(x)
        
    #     # Global Pooling (Average over sequence, or take CLS token if we had one. Simple Avg here.)
    #     # Assuming padding tokens might affect avg, but let's effectively mean pool.
    #     x_pool = x.mean(dim=1) 
    #     # ##EMİN## it is better to use cls token
    #     # take CLS token
    #     #x_pool = x[:, 0, :]    
        
    #     out = self.regressor(x_pool)
    #     return out.squeeze()
    def forward(self, src):
        # src: [batch_size, seq_len] (Örn: [[5, 2, 9, 0, 0], ...])
        
        # --- 1. MASK HESAPLAMA (YENİ KISIM) ---
        # Padding olan yerler True, dolu yerler False olsun
        # src_mask: [batch, seq_len] -> Örn: [False, False, False, True, True]
        padding_mask = (src == self.pad_idx)
        
        # CLS Token Maskesi:
        # CLS token asla padding değildir, o yüzden False (Maskeleme) diyoruz.
        batch_size = src.size(0)
        cls_mask = torch.zeros((batch_size, 1), device=src.device, dtype=torch.bool)
        
        # İkisini birleştir: [batch, seq_len + 1]
        # Örn: [False (CLS), False, False, False, True, True]
        full_mask = torch.cat((cls_mask, padding_mask), dim=1)

        # --- 2. EMBEDDING & CLS ---
        x = self.embedding(src)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # --- 3. POSITIONAL ENCODING ---
        curr_len = x.size(1)
        x = x + self.pos_encoder[:, :curr_len, :]

        # --- 4. TRANSFORMER (Maskeyi buraya veriyoruz) ---
        # Transformer artık True olan yerlere (padding) attention uygulamayacak (-inf basacak).
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        
        # --- 5. POOLING ---
        x_pool = x[:, 0, :] # Sadece CLS token
        
        return self.regressor(x_pool).squeeze()