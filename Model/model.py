
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SurrogateModel(nn.Module):
    def __init__(self, d_model=256, num_head=8):
        super(SurrogateModel, self).__init__()
        self.d_model = d_model
        self.encoder = Encoder(d_model=d_model, num_heads=num_head)

        self.sru_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model)
        )

        self.ln_q_self = nn.LayerNorm(d_model)
        self.sru_self_attn = nn.MultiheadAttention(d_model, num_head, batch_first=True)
        
        self.ln_q_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_head, batch_first=True)
        
        self.ln_ffn = nn.LayerNorm(d_model)
        self.sru_ffn = nn.Sequential(
            nn.Linear(d_model, d_model *2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(0.05)
        )

        # MLP de sortie 
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def encode_drift(self, drift):
        return self.encoder(drift)

    def forward(self, drift, srus, drift_tokens=None):
        # drift: (B, N, T, 2) | srus: (B, S, N_sru, 5)
        B, S, N_sru, _ = srus.shape
        
        if drift_tokens == None:
            drift_tokens = self.encoder(drift)      # [B, T, d_model]
        
        sru_feat = srus[..., :4].reshape(B*S, N_sru, 4)
        sru_mask = (srus[..., 4] == 0).reshape(B*S, N_sru)
        queries = self.sru_embed(sru_feat)
        
        keys = drift_tokens.unsqueeze(1).expand(-1, S, -1, -1).reshape(B*S, -1, self.d_model)
        
        mask_float = (~sru_mask).float().unsqueeze(-1)
        
        q_norm = self.ln_q_self(queries)
        res_self, _ = self.sru_self_attn(q_norm, q_norm, q_norm, key_padding_mask=sru_mask)
        queries = (queries + res_self) * mask_float
        
        q_norm = self.ln_q_cross(queries)
        res_cross, _ = self.cross_attn(q_norm, keys, keys)
        queries = (queries + res_cross) * mask_float

        q_norm = self.ln_ffn(queries)
        queries = (queries + self.sru_ffn(q_norm)) * mask_float

        out = queries.sum(dim=1)                 # [B*S, d_model]
        out = self.mlp_head(out)                 # [B*S, 1]
        
        return out.reshape(B, S)

class Encoder(nn.Module):
    def __init__(self, d_model=256, num_heads=8, num_layers=3):
        super().__init__()
        self.d_model = d_model
        self.pointnet = VanillaPointNet(d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model*4,
            dropout=0.05, activation='gelu', batch_first=True, norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x input: [B, N, T, 2]
        B, N, T, C = x.shape
        
        x = x.transpose(1, 2).reshape(B * T, N, C)
        
        x = self.pointnet(x)                # [B*T, d_model]
        
        x = x.reshape(B, T, self.d_model)   # [B, T, 256]
        
        x = self.pos_encoding(x)
        x = self.temporal_transformer(x)    
        
        return self.final_norm(x)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
    
class VanillaPointNet(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        
        self.local_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256)
        )
        
        self.global_mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # x: [B*T, N, 2]
        
        x = self.local_mlp(x)             # [B*T, N, 256]
        
        x, _ = torch.max(x, dim=1)        # [B*T, 256]
        
        x = self.global_mlp(x)            # [B*T, d_model]
        
        return x
    
