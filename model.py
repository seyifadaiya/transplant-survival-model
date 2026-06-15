
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



# FEATURE TOKENIZER ; converts tabular features into embeddings
class FeatureTokenizer(nn.Module):
    """
    Converts raw tabular features into a sequence of d_model-dimensional tokens
    Each feature gets its own learned linear projection + positional embedding
    
    """
    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Each feature gets its own projection to d_model dimensions
        # like a per-feature embedding layer
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        
        # Learnable positional embeddings for feature ordering
        self.position_embeddings = nn.Parameter(
            torch.randn(1, n_features, d_model) * 0.02
        )
        
        # Layer norm after tokenization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
       
        batch_size = x.size(0)
        tokens = []
        
        for i in range(self.n_features):
            # Project each scalar feature to d_model dimensions
            feat_val = x[:, i:i+1]  # (batch_size, 1)
            token = self.feature_projections[i](feat_val)  # (batch_size, d_model)
            tokens.append(token)
        
        # Stack into sequence: (batch_size, n_features, d_model)
        tokens = torch.stack(tokens, dim=1)
        
        # Add positional embeddings
        tokens = tokens + self.position_embeddings
        
        return self.layer_norm(tokens)



# TRANSFORMER ENCODER BLOCK
class TransformerEncoderBlock(nn.Module):
    """
    Standard transformer encoder with multi-head self-attention + FFN
    for learning within-group feature interactions
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
       
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x



# CROSS-ATTENTION BLOCK 
class CrossAttentionBlock(nn.Module):
    """
    Cross-attention between donor and recipient feature embeddings
    
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Donor attends to Recipient
        self.donor_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Recipient attends to Donor
        self.recip_cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # FFN for each stream
        self.donor_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.recip_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        
        # Layer norms
        self.donor_norm1 = nn.LayerNorm(d_model)
        self.donor_norm2 = nn.LayerNorm(d_model)
        self.recip_norm1 = nn.LayerNorm(d_model)
        self.recip_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, donor_tokens, recip_tokens):
       
        # Donor attends to Recipient (Query=Donor, Key/Value=Recipient)
        d_cross, self.donor_attn_weights = self.donor_cross_attn(
            donor_tokens, recip_tokens, recip_tokens
        )
        donor_out = self.donor_norm1(donor_tokens + self.dropout(d_cross))
        donor_out = self.donor_norm2(donor_out + self.donor_ffn(donor_out))
        
        # Recipient attends to Donor (Query=Recipient, Key/Value=Donor)
        r_cross, self.recip_attn_weights = self.recip_cross_attn(
            recip_tokens, donor_tokens, donor_tokens
        )
        recip_out = self.recip_norm1(recip_tokens + self.dropout(r_cross))
        recip_out = self.recip_norm2(recip_out + self.recip_ffn(recip_out))
        
        return donor_out, recip_out



# COMPETING RISKS OUTPUT HEAD - DEEPHIT-STYLE 
class CompetingRisksHead(nn.Module):
   
    def __init__(self, d_input, num_risks, num_time_bins, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins
        
        # Shared projection
        self.shared_fc = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Cause-specific sub-networks
        self.cause_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_time_bins)
            ) for _ in range(num_risks)
        ])
    
    def forward(self, z):
        
        h = self.shared_fc(z)
        
        # Get cause-specific logits
        cause_logits = []
        for cause_net in self.cause_nets:
            cause_logits.append(cause_net(h))  # (batch_size, num_time_bins)
        
        # Stack: (batch_size, num_risks * num_time_bins)
        all_logits = torch.cat(cause_logits, dim=-1)
        
        # Softmax over ALL (risk × time) combinations
        # This ensures sum of all P(T=t, K=k) = 1
        all_probs = F.softmax(all_logits, dim=-1)
        
        # Reshape to (batch_size, num_risks, num_time_bins)
        pmf = all_probs.view(-1, self.num_risks, self.num_time_bins)
        
        # Compute overall survival: S(t) = 1 - sum_k CIF_k(t)
        # CIF_k(t) = sum_{s<=t} P(T=s, K=k)
        cif = torch.cumsum(pmf, dim=-1)  # (batch_size, num_risks, num_time_bins)
        total_cif = cif.sum(dim=1)  # (batch_size, num_time_bins)
        surv = 1.0 - total_cif
        
        return pmf, cif, surv



# FULL MODEL - CACRT
class CACRT(nn.Module):
    
    def __init__(
        self,
        n_donor_features,
        n_recip_features,
        n_match_features=0,
        d_model=64,
        n_heads=4,
        n_self_layers=2,
        n_cross_layers=2,
        d_ff=128,
        num_risks=2,
        num_time_bins=100,
        dropout=0.1
    ):
        super().__init__()
        
        self.n_donor_features = n_donor_features
        self.n_recip_features = n_recip_features
        self.n_match_features = n_match_features
        self.num_risks = num_risks
        self.num_time_bins = num_time_bins
        
        # Feature Tokenizers 
        self.donor_tokenizer = FeatureTokenizer(n_donor_features, d_model)
        self.recip_tokenizer = FeatureTokenizer(n_recip_features, d_model)
        
        # match features (HLA mismatches, etc.) get a separate encoder
        if n_match_features > 0:
            self.match_tokenizer = FeatureTokenizer(n_match_features, d_model)
            self.match_encoder = nn.ModuleList([
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_self_layers)
            ])
        
        # Self-Attention Encoders (within-group) 
        self.donor_encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_self_layers)
        ])
        self.recip_encoder = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_self_layers)
        ])
        
        # Cross-Attention Layers (donor ↔ recipient)
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_cross_layers)
        ])
        
        # Fusion 
        # Pool each stream (mean pooling) and concatenate
        total_features = n_donor_features + n_recip_features
        if n_match_features > 0:
            total_features += n_match_features
        
        fusion_input_dim = d_model * (2 + (1 if n_match_features > 0 else 0))
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Competing Risks Output Head
        self.output_head = CompetingRisksHead(
            d_input=d_model * 2,
            num_risks=num_risks,
            num_time_bins=num_time_bins,
            hidden_dim=d_model * 2,
            dropout=dropout
        )
    
    def forward(self, x_donor, x_recip, x_match=None):
        
        # Tokenize features
        donor_tokens = self.donor_tokenizer(x_donor)   # (B, n_d, d_model)
        recip_tokens = self.recip_tokenizer(x_recip)   # (B, n_r, d_model)
        
        # Self-attention within each group
        for layer in self.donor_encoder:
            donor_tokens = layer(donor_tokens)
        for layer in self.recip_encoder:
            recip_tokens = layer(recip_tokens)
        
        # Cross-attention between donor and recipient
        for cross_layer in self.cross_attention_layers:
            donor_tokens, recip_tokens = cross_layer(donor_tokens, recip_tokens)
        
        # Pool each stream (mean over feature tokens)
        donor_pooled = donor_tokens.mean(dim=1)   # (B, d_model)
        recip_pooled = recip_tokens.mean(dim=1)   # (B, d_model)
        
        # Handle match features if present
        if x_match is not None and self.n_match_features > 0:
            match_tokens = self.match_tokenizer(x_match)
            for layer in self.match_encoder:
                match_tokens = layer(match_tokens)
            match_pooled = match_tokens.mean(dim=1)
            fused = torch.cat([donor_pooled, recip_pooled, match_pooled], dim=-1)
        else:
            fused = torch.cat([donor_pooled, recip_pooled], dim=-1)
        
        # Fusion layer
        z = self.fusion(fused)  # (B, d_model * 2)
        
        # Competing risks output
        pmf, cif, surv = self.output_head(z)
        
        return pmf, cif, surv
    
    def get_cross_attention_weights(self):
        """
        Extract cross-attention weights for interpretability
        Returns the attention maps from all cross-attention layers
        """
        weights = []
        for layer in self.cross_attention_layers:
            weights.append({
                'donor_to_recip': layer.donor_attn_weights.detach().cpu(),
                'recip_to_donor': layer.recip_attn_weights.detach().cpu()
            })
        return weights
