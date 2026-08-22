# -*- coding: utf-8 -*-
"""
Gradient Ascent Optimization for Yield Strength Maximization

This script uses a trained model to find the optimal composition
that maximizes yield strength while keeping the processing text fixed.

Requirements:
    - complete_model_seed_XXX.pt (trained model with all parameters)
    - process.txt (fixed processing text, first row first column)
    - composition.csv (initial composition, 103 columns)
    - ./pretrained/ (BERT model for text embedding)

Output:
    - itera.csv (all iterations with 103 columns + predicted YS)
    - Console output with final optimal composition and maximum YS
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List, Dict, Tuple
from tqdm import tqdm
from datetime import datetime

# Hugging Face libraries
from transformers import AutoModel, AutoTokenizer

# --- Configuration ---
CONFIG = {
    # File paths
    "model_path": "./outputs/reg_model_saved/complete_model_seed_123.pt",
    "bert_path": "./pretrained",
    "process_text_path": "./process.txt",
    "composition_path": "./composition.csv",
    "output_path": "./itera.csv",
    
    # Optimization parameters
    "learning_rate": 0.01,
    "max_iterations": 1000,
    "convergence_threshold": 0.01,  # MPa
    "convergence_patience": 10,     # Stop if no improvement for this many iterations
    "print_every": 50,              # Print progress every N iterations
    
    # Composition constraint
    "composition_sum": 100.0,       # Target sum for non-zero elements
}

# --- Activation layers (same as training code) ---
ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "relu": nn.ReLU,
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh
}

# --- Model Components (copied from training code) ---

class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention mechanism for feature fusion."""
    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class FcUnitWithResidual(nn.Module):
    """MLP block with residual connection."""
    def __init__(self, in_features: int, out_features: int, normalization: str = "layer_norm", 
                 activation: str = 'relu', dropout_prob: float = 0.1):
        super().__init__()
        if normalization == "layer_norm": 
            self.norm = nn.LayerNorm(in_features)
        elif normalization == "batch_norm": 
            self.norm = nn.BatchNorm1d(in_features)
        else: 
            self.norm = nn.Identity()
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)
        
        self.use_residual = (in_features == out_features)
        if not self.use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = None
        
    def forward(self, x):
        residual = x
        out = self.dropout(self.act_fn(self.fc(self.norm(x))))
        
        if self.use_residual:
            out = out + residual
        elif self.residual_proj is not None:
            out = out + self.residual_proj(residual)
            
        return out


class Cnn1dUnitWithResidual(nn.Module):
    """1D CNN block with residual."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 activation: str = 'relu', pooling: str = 'max', pool_kernel_size: int = 2):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, padding=(kernel_size // 2))
        self.act_fn = ALL_ACT_LAYERS[activation]()
        if pooling == "max": 
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        else: 
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        out = self.pool(self.act_fn(self.conv1d(x)))
        
        if self.use_residual:
            residual_pooled = self.pool(residual)
            if out.shape == residual_pooled.shape:
                out = out + residual_pooled
        else:
            residual = self.residual_conv(residual)
            residual_pooled = self.pool(residual)
            if out.shape == residual_pooled.shape:
                out = out + residual_pooled
                
        return out


class Cnn2dUnitWithResidual(nn.Module):
    """2D CNN block with residual."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 activation: str = 'relu', pooling: str = 'max', pool_kernel_size: int = 2):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, padding=(kernel_size // 2))
        self.act_fn = ALL_ACT_LAYERS[activation]()
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size)
            
        self.use_residual = (in_channels == out_channels)
        if not self.use_residual:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.act_fn(self.conv2d(x))
        
        if out.shape[2] >= self.pool.kernel_size and out.shape[3] >= self.pool.kernel_size:
            out = self.pool(out)
            
            if self.use_residual:
                residual_pooled = self.pool(residual)
                if out.shape == residual_pooled.shape:
                    out = out + residual_pooled
            else:
                residual = self.residual_conv(residual)
                residual_pooled = self.pool(residual)
                if out.shape == residual_pooled.shape:
                    out = out + residual_pooled
        
        return out


class CustomAdvancedModel(nn.Module):
    """Enhanced multimodal regression model with attention-based fusion."""
    def __init__(
            self, simple_layer_list: List[int], concat_layer_list: List[int],
            seq_embed_con1d_list: List[int], seq_embed_fc_list: List[int],
            seq_embed_con2d_list: List[int], seq_embed_2d_fc_list: List[int],
            simple_layer_drop_prob: float = 0.0, concat_layer_drop_prob: float = 0.0,
            use_attention_fusion: bool = True, num_attention_heads: int = 8,
            num_chem_features: int = 7
    ):
        super().__init__()
        
        self.use_attention_fusion = use_attention_fusion
        
        self.simple_layer = self._create_mlp_pathway_with_residual([768] + simple_layer_list, simple_layer_drop_prob)
        self.seq_embed_layer = self._create_1d_cnn_pathway_with_residual(1, seq_embed_con1d_list, seq_embed_fc_list, simple_layer_drop_prob)
        self.common_embed_layer = self._create_2d_cnn_pathway_with_residual(1, seq_embed_con2d_list, seq_embed_2d_fc_list, simple_layer_drop_prob)
        
        if num_chem_features > 0:
            self.chem_feature_processor = nn.Sequential(
                nn.Linear(num_chem_features, 64),
                nn.ReLU(),
                nn.Dropout(simple_layer_drop_prob),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(simple_layer_drop_prob),
                nn.Linear(128, 64)
            )
        else:
            self.chem_feature_processor = None
        
        if use_attention_fusion:
            self.projection_dim = 256
            self.com_projection = nn.LazyLinear(self.projection_dim)
            self.text_projection = nn.LazyLinear(self.projection_dim)
            self.cross_projection = nn.LazyLinear(self.projection_dim)
            if num_chem_features > 0:
                self.chem_projection = nn.Linear(64, self.projection_dim)
            
            self.attention_fusion = MultiHeadAttentionFusion(
                input_dim=self.projection_dim,
                num_heads=num_attention_heads,
                dropout=concat_layer_drop_prob
            )
            
            self.final_layers = nn.Sequential(
                nn.LazyLinear(concat_layer_list[0]),
                nn.ReLU(),
                nn.Dropout(concat_layer_drop_prob),
                *[nn.Sequential(
                    nn.Linear(concat_layer_list[i], concat_layer_list[i+1]),
                    nn.ReLU(),
                    nn.Dropout(concat_layer_drop_prob)
                ) for i in range(len(concat_layer_list)-1)],
                nn.Linear(concat_layer_list[-1], 1)
            )
        else:
            self.concat_layer = nn.Sequential(
                nn.LazyLinear(concat_layer_list[0]), nn.ReLU(), nn.Dropout(concat_layer_drop_prob),
                *[nn.Sequential(
                    nn.Linear(concat_layer_list[i], concat_layer_list[i+1]), nn.ReLU(), nn.Dropout(concat_layer_drop_prob)
                  ) for i in range(len(concat_layer_list)-1)],
                nn.Linear(concat_layer_list[-1], 1)
            )

    def _create_mlp_pathway_with_residual(self, layer_dims: List[int], dropout: float):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(FcUnitWithResidual(layer_dims[i], layer_dims[i+1], dropout_prob=dropout))
        return nn.Sequential(*layers)

    def _create_1d_cnn_pathway_with_residual(self, in_channels, conv1d_list, fc_list, dropout):
        layers = []
        current_channels = in_channels
        for out_channels in conv1d_list:
            layers.append(Cnn1dUnitWithResidual(current_channels, out_channels))
            current_channels = out_channels
        
        layers.extend([nn.Flatten(), nn.LazyLinear(fc_list[0])])
        for i in range(len(fc_list) - 1):
            layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_list[i], fc_list[i+1])])
        return nn.Sequential(*layers)

    def _create_2d_cnn_pathway_with_residual(self, in_channels, conv2d_list, fc_list, dropout):
        layers = []
        current_channels = in_channels
        for out_channels in conv2d_list:
            layers.append(Cnn2dUnitWithResidual(current_channels, out_channels))
            current_channels = out_channels
        
        layers.extend([nn.Flatten(), nn.LazyLinear(fc_list[0])])
        for i in range(len(fc_list) - 1):
            layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_list[i], fc_list[i+1])])
        return nn.Sequential(*layers)

    def forward(self, eles, chem_features, text_embeds, com_embeds, labels=None):
        com_dense = self.simple_layer(com_embeds)
        text_dense = self.simple_layer(text_embeds)

        com_1d_out = self.seq_embed_layer(com_dense.unsqueeze(1))
        text_1d_out = self.seq_embed_layer(text_dense.unsqueeze(1))
        
        cross_modal_in = torch.stack([com_embeds, text_embeds], dim=1).unsqueeze(1)
        cross_modal_2d_out = self.common_embed_layer(cross_modal_in)
        
        if self.chem_feature_processor is not None and chem_features.shape[1] > 0:
            chem_processed = self.chem_feature_processor(chem_features)
        else:
            chem_processed = None

        if self.use_attention_fusion:
            com_proj = self.com_projection(com_1d_out)
            text_proj = self.text_projection(text_1d_out)
            cross_proj = self.cross_projection(cross_modal_2d_out)
            
            features = [com_proj.unsqueeze(1), text_proj.unsqueeze(1), cross_proj.unsqueeze(1)]
            if chem_processed is not None:
                chem_proj = self.chem_projection(chem_processed)
                features.append(chem_proj.unsqueeze(1))
            
            stacked_features = torch.cat(features, dim=1)
            fused_features = self.attention_fusion(stacked_features)
            fused_flat = fused_features.view(fused_features.size(0), -1)
            return self.final_layers(fused_flat)
        else:
            if chem_processed is not None:
                combined = torch.cat([cross_modal_2d_out, com_1d_out, text_1d_out, chem_processed], dim=1)
            else:
                combined = torch.cat([cross_modal_2d_out, com_1d_out, text_1d_out], dim=1)
            return self.concat_layer(combined)


# --- Utility Functions ---

def cls_pooling(model_output):
    """Extract the CLS token embedding from BERT output."""
    return model_output.last_hidden_state[:, 0]


def get_text_embedding(text: str, tokenizer, bert_model, device: torch.device) -> torch.Tensor:
    """Convert text to BERT embedding."""
    bert_model.eval()
    with torch.no_grad():
        encoded_input = tokenizer(
            text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
        )
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        model_output = bert_model(**encoded_input)
        embedding = cls_pooling(model_output)
    return embedding  # Shape: [1, 768]


def compute_com_embeds(compositions: torch.Tensor, element_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute composition embeddings from compositions and element embeddings.
    
    Args:
        compositions: [1, n_elements] tensor of composition values
        element_embeddings: [n_elements, 768] tensor of BERT embeddings for elements
    
    Returns:
        [1, 768] tensor of weighted composition embedding
    """
    # Weighted average of element embeddings
    weights = compositions / (compositions.sum(dim=1, keepdim=True) + 1e-9)  # Normalize
    com_embeds = torch.matmul(weights, element_embeddings)  # [1, 768]
    return com_embeds


def compute_chem_features_differentiable(
    compositions: torch.Tensor, 
    num_elements: int,
    chem_norm_params: Dict,
    device: torch.device
) -> torch.Tensor:
    """
    Compute chemical features from compositions in a differentiable way.
    
    Args:
        compositions: [1, n_elements] tensor of composition values
        num_elements: Number of non-zero elements (constant)
        chem_norm_params: Normalization parameters for chemical features
        device: Torch device
    
    Returns:
        [1, 7] tensor of normalized chemical features
    """
    eps = 1e-10
    
    # Ensure non-negative
    comp = torch.clamp(compositions, min=0)
    
    # 1. num_elements (constant, since we fix which elements are active)
    num_elem = torch.tensor([[float(num_elements)]], device=device)
    
    # 2. composition_entropy: -Σ(p_i * log(p_i))
    comp_sum = comp.sum(dim=1, keepdim=True) + eps
    p = comp / comp_sum
    p_safe = torch.clamp(p, min=eps)
    entropy = -(p_safe * torch.log(p_safe)).sum(dim=1, keepdim=True)
    
    # 3. max_fraction (soft approximation using softmax)
    temperature = 0.1
    soft_weights = F.softmax(comp / temperature, dim=1)
    max_frac = (comp * soft_weights).sum(dim=1, keepdim=True)
    
    # 4. min_fraction (soft approximation using negative softmax)
    # Only consider non-zero elements - use large negative for zeros
    comp_for_min = torch.where(comp > eps, comp, torch.tensor(1e6, device=device))
    soft_weights_min = F.softmax(-comp_for_min / temperature, dim=1)
    min_frac = (comp * soft_weights_min).sum(dim=1, keepdim=True)
    
    # 5. composition_variance
    mean_comp = comp.mean(dim=1, keepdim=True)
    variance = ((comp - mean_comp) ** 2).mean(dim=1, keepdim=True)
    
    # 6. composition_std
    std = torch.sqrt(variance + eps)
    
    # 7. gini_coefficient (approximation)
    # Gini = (2 * Σ(i * x_sorted[i])) / (n * Σx) - (n+1)/n
    # Use soft sorting approximation
    n = comp.shape[1]
    sorted_comp, _ = torch.sort(comp, dim=1)
    indices = torch.arange(1, n + 1, dtype=torch.float32, device=device).unsqueeze(0)
    numerator = 2 * (indices * sorted_comp).sum(dim=1, keepdim=True)
    denominator = n * (sorted_comp.sum(dim=1, keepdim=True) + eps)
    gini = numerator / denominator - (n + 1) / n
    
    # Stack all features
    features = torch.cat([num_elem, entropy, max_frac, min_frac, variance, std, gini], dim=1)
    
    # Normalize using saved parameters
    feature_names = ['num_elements', 'composition_entropy', 'max_fraction', 
                     'min_fraction', 'composition_variance', 'composition_std', 'gini_coefficient']
    
    normalized_features = []
    for i, name in enumerate(feature_names):
        if name in chem_norm_params:
            mean = chem_norm_params[name]['mean']
            std_val = chem_norm_params[name]['std']
            if std_val > 0:
                normalized_features.append((features[:, i:i+1] - mean) / std_val)
            else:
                normalized_features.append(features[:, i:i+1] - mean)
        else:
            normalized_features.append(features[:, i:i+1])
    
    return torch.cat(normalized_features, dim=1)


def load_complete_model(model_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load the complete model with all saved parameters.
    
    Returns:
        Tuple of (model, saved_dict)
    """
    print(f"Loading model from {model_path}...")
    # Note: weights_only=False is required because we saved numpy arrays (element_embeddings)
    # This is safe since you trained this model yourself
    saved_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    config = saved_dict['config']
    
    # Rebuild model architecture
    model = CustomAdvancedModel(
        simple_layer_list=config['simple_layer_list'],
        concat_layer_list=config['concat_layer_list'],
        seq_embed_con1d_list=config['seq_embed_con1d_list'],
        seq_embed_fc_list=config['seq_embed_fc_list'],
        seq_embed_con2d_list=config['seq_embed_con2d_list'],
        seq_embed_2d_fc_list=config['seq_embed_2d_fc_list'],
        simple_layer_drop_prob=config.get('simple_layer_drop_prob', 0.1),
        concat_layer_drop_prob=config.get('concat_layer_drop_prob', 0.1),
        use_attention_fusion=config.get('use_attention_fusion', True),
        num_attention_heads=config.get('num_attention_heads', 8),
        num_chem_features=7
    ).to(device)
    
    # Initialize lazy layers with dummy forward pass
    dummy_input = {
        'eles': torch.zeros(1, 103).to(device),
        'chem_features': torch.zeros(1, 7).to(device),
        'text_embeds': torch.zeros(1, 768).to(device),
        'com_embeds': torch.zeros(1, 768).to(device),
    }
    with torch.no_grad():
        model(**dummy_input)
    
    # Load weights
    model_state = saved_dict['model_state_dict']
    
    # Handle SWA model state dict (remove 'module.' prefix if present)
    if any(k.startswith('module.') for k in model_state.keys()):
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    
    # Try to load state dict
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, trying non-strict: {e}")
        model.load_state_dict(model_state, strict=False)
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, saved_dict


def load_process_text(file_path: str) -> str:
    """Load processing text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.readline().strip()
    return text


def load_composition(file_path: str, expected_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Load composition from CSV file.

    Args:
        file_path: Path to the composition CSV file.
        expected_cols: Optional list of element column names expected by the model.
                       If provided, the CSV will be aligned to these columns (missing
                       columns filled with 0, extra columns dropped).

    Returns:
        Tuple of (dataframe, column_names, non_zero_indices)
    """
    df = pd.read_csv(file_path)

    if expected_cols is not None:
        # Align CSV columns to the model's expected element columns
        aligned = pd.DataFrame(columns=expected_cols)
        for col in expected_cols:
            if col in df.columns:
                aligned[col] = df[col].values
            else:
                aligned[col] = 0.0
        df = aligned

    column_names = df.columns.tolist()

    # Get the first row values
    values = df.iloc[0].values.astype(float)

    # Find non-zero indices
    non_zero_indices = [i for i, v in enumerate(values) if v > 0]

    return df, column_names, non_zero_indices


class CompositionOptimizer:
    """
    Gradient ascent optimizer for composition optimization.
    Uses softmax parameterization to ensure sum constraint.
    """
    
    def __init__(
        self,
        model: nn.Module,
        text_embeds: torch.Tensor,
        element_embeddings: torch.Tensor,
        chem_norm_params: Dict,
        yield_norm_params: Dict,
        initial_composition: np.ndarray,
        non_zero_indices: List[int],
        column_names: List[str],
        device: torch.device,
        learning_rate: float = 0.1,
        composition_sum: float = 100.0
    ):
        self.model = model
        self.model.eval()  # Keep in eval mode
        
        self.text_embeds = text_embeds  # Fixed [1, 768]
        self.element_embeddings = element_embeddings  # [n_elements, 768]
        self.chem_norm_params = chem_norm_params
        self.yield_norm_params = yield_norm_params
        self.device = device
        self.composition_sum = composition_sum
        self.column_names = column_names
        self.non_zero_indices = non_zero_indices
        self.n_elements = len(initial_composition)
        self.n_active = len(non_zero_indices)
        
        # Initialize raw parameters (unconstrained) for non-zero elements
        # Start with values that will give similar proportions after softmax
        initial_active = initial_composition[non_zero_indices]
        initial_active = np.maximum(initial_active, 1e-6)  # Avoid zeros
        
        # Initialize as log of proportions (inverse of softmax)
        initial_proportions = initial_active / initial_active.sum()
        initial_raw = np.log(initial_proportions + 1e-10)
        
        self.raw_params = torch.tensor(
            initial_raw, dtype=torch.float32, device=device, requires_grad=True
        )
        
        self.optimizer = Adam([self.raw_params], lr=learning_rate)
        
        # Store zero element indices
        self.zero_indices = [i for i in range(self.n_elements) if i not in non_zero_indices]
        
        print(f"\nOptimizer initialized:")
        print(f"  Total elements: {self.n_elements}")
        print(f"  Active elements: {self.n_active}")
        print(f"  Active element names: {[column_names[i] for i in non_zero_indices]}")
    
    def get_composition(self) -> torch.Tensor:
        """Convert raw parameters to valid composition."""
        # Softmax to get proportions (sum to 1, all positive)
        proportions = F.softmax(self.raw_params, dim=0)
        
        # Scale to target sum
        active_composition = proportions * self.composition_sum
        
        # Build full composition tensor
        full_composition = torch.zeros(self.n_elements, device=self.device)
        for i, idx in enumerate(self.non_zero_indices):
            full_composition[idx] = active_composition[i]
        
        return full_composition.unsqueeze(0)  # [1, n_elements]
    
    def predict_yield_strength(self, composition: torch.Tensor) -> torch.Tensor:
        """
        Predict yield strength from composition.
        
        Args:
            composition: [1, n_elements] tensor
        
        Returns:
            Predicted yield strength (denormalized, in MPa)
        """
        # Compute composition embeddings
        com_embeds = compute_com_embeds(composition, self.element_embeddings)
        
        # Compute chemical features
        chem_features = compute_chem_features_differentiable(
            composition, self.n_active, self.chem_norm_params, self.device
        )
        
        # Forward pass through model
        pred_normalized = self.model(
            eles=composition,
            chem_features=chem_features,
            text_embeds=self.text_embeds,
            com_embeds=com_embeds
        )
        
        # Denormalize prediction
        mean = self.yield_norm_params['mean']
        std = self.yield_norm_params['std']
        pred_denorm = pred_normalized * std + mean
        
        return pred_denorm
    
    def step(self) -> Tuple[float, np.ndarray]:
        """
        Perform one optimization step.
        
        Returns:
            Tuple of (predicted_yield_strength, current_composition)
        """
        self.optimizer.zero_grad()
        
        # Get current composition
        composition = self.get_composition()
        
        # Predict yield strength
        yield_strength = self.predict_yield_strength(composition)
        
        # Gradient ASCENT: maximize yield strength
        loss = -yield_strength
        loss.backward()
        
        self.optimizer.step()
        
        return yield_strength.item(), composition.detach().cpu().numpy().flatten()
    
    def get_current_state(self) -> Tuple[float, np.ndarray]:
        """Get current yield strength and composition without updating."""
        with torch.no_grad():
            composition = self.get_composition()
            yield_strength = self.predict_yield_strength(composition)
        return yield_strength.item(), composition.cpu().numpy().flatten()


def run_optimization(config: Dict) -> None:
    """Main optimization function."""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load all components ---
    print("\n" + "="*60)
    print("LOADING COMPONENTS")
    print("="*60)
    
    # 1. Load trained model
    model, saved_dict = load_complete_model(config['model_path'], device)
    norm_params = saved_dict['normalization_params']
    
    # 2. Load BERT model
    print(f"\nLoading BERT model from {config['bert_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['bert_path'])
    bert_model = AutoModel.from_pretrained(config['bert_path']).to(device)
    bert_model.eval()
    print("BERT model loaded successfully!")
    
    # 3. Load and encode processing text
    print(f"\nLoading processing text from {config['process_text_path']}...")
    process_text = load_process_text(config['process_text_path'])
    print(f"Processing text: \"{process_text[:100]}{'...' if len(process_text) > 100 else ''}\"")
    
    text_embeds = get_text_embedding(process_text, tokenizer, bert_model, device)
    print(f"Text embedding shape: {text_embeds.shape}")
    
    # 4. Load composition (align to model's expected element columns)
    print(f"\nLoading composition from {config['composition_path']}...")
    expected_element_cols = norm_params.get('element_cols', None)
    comp_df, column_names, non_zero_indices = load_composition(
        config['composition_path'], expected_cols=expected_element_cols
    )
    initial_composition = comp_df.iloc[0].values.astype(float)
    
    print(f"Total columns: {len(column_names)}")
    print(f"Non-zero elements ({len(non_zero_indices)}):")
    for idx in non_zero_indices:
        print(f"  {column_names[idx]}: {initial_composition[idx]:.2f}%")
    print(f"Initial sum: {initial_composition.sum():.2f}%")
    
    # 5. Get element embeddings
    element_embeddings = torch.tensor(
        norm_params['element_embeddings'], dtype=torch.float32, device=device
    )
    print(f"\nElement embeddings shape: {element_embeddings.shape}")
    
    # --- Initialize optimizer ---
    print("\n" + "="*60)
    print("INITIALIZING OPTIMIZER")
    print("="*60)
    
    optimizer = CompositionOptimizer(
        model=model,
        text_embeds=text_embeds,
        element_embeddings=element_embeddings,
        chem_norm_params=norm_params.get('chem_features', {}),
        yield_norm_params=norm_params.get('yield_strength', {'mean': 0, 'std': 1}),
        initial_composition=initial_composition,
        non_zero_indices=non_zero_indices,
        column_names=column_names,
        device=device,
        learning_rate=config['learning_rate'],
        composition_sum=config['composition_sum']
    )
    
    # --- Run optimization ---
    print("\n" + "="*60)
    print("STARTING GRADIENT ASCENT OPTIMIZATION")
    print("="*60)
    
    # Get initial state
    initial_ys, initial_comp = optimizer.get_current_state()
    print(f"\nInitial Yield Strength: {initial_ys:.2f} MPa")
    print(f"Initial composition sum: {initial_comp.sum():.2f}%")
    
    # History storage
    history = []
    history.append({
        'step': 0,
        'yield_strength': initial_ys,
        'composition': initial_comp.copy()
    })
    
    best_ys = initial_ys
    best_composition = initial_comp.copy()
    patience_counter = 0
    
    print(f"\nRunning optimization (max {config['max_iterations']} iterations)...")
    print(f"Convergence: YS improvement < {config['convergence_threshold']} MPa for {config['convergence_patience']} iterations")
    print("-" * 60)
    
    for step in range(1, config['max_iterations'] + 1):
        # Optimization step
        current_ys, current_comp = optimizer.step()
        
        # Record history
        history.append({
            'step': step,
            'yield_strength': current_ys,
            'composition': current_comp.copy()
        })
        
        # Check improvement
        improvement = current_ys - best_ys
        
        if improvement > config['convergence_threshold']:
            best_ys = current_ys
            best_composition = current_comp.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if step % config['print_every'] == 0 or step == 1:
            print(f"Step {step:4d}: YS = {current_ys:8.2f} MPa, "
                  f"Sum = {current_comp.sum():6.2f}%, "
                  f"Improvement = {improvement:+.4f} MPa")
        
        # Check convergence
        if patience_counter >= config['convergence_patience']:
            print(f"\nConverged at step {step} (no improvement for {config['convergence_patience']} iterations)")
            break
    
    else:
        print(f"\nReached maximum iterations ({config['max_iterations']})")
    
    # --- Output results ---
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    print(f"\nInitial Yield Strength: {initial_ys:.2f} MPa")
    print(f"Final Yield Strength:   {best_ys:.2f} MPa")
    print(f"Improvement:            {best_ys - initial_ys:.2f} MPa ({(best_ys - initial_ys) / initial_ys * 100:.1f}%)")
    
    print(f"\nOptimal composition:")
    for idx in non_zero_indices:
        initial_val = initial_composition[idx]
        final_val = best_composition[idx]
        change = final_val - initial_val
        print(f"  {column_names[idx]:4s}: {final_val:6.2f}% (initial: {initial_val:6.2f}%, change: {change:+6.2f}%)")
    
    print(f"\nComposition sum: {best_composition.sum():.2f}%")
    
    # --- Save to CSV ---
    print(f"\nSaving iteration history to {config['output_path']}...")
    
    # Build DataFrame
    output_data = []
    for record in history:
        row = {'Step': record['step'], 'Predicted_YS_MPa': record['yield_strength']}
        for i, col_name in enumerate(column_names):
            row[col_name] = record['composition'][i]
        output_data.append(row)
    
    output_df = pd.DataFrame(output_data)
    
    # Reorder columns: Step, element columns, Predicted_YS
    cols = ['Step'] + column_names + ['Predicted_YS_MPa']
    output_df = output_df[cols]
    
    output_df.to_csv(config['output_path'], index=False)
    print(f"Saved {len(history)} iterations to {config['output_path']}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\nMaximum Yield Strength: {best_ys:.2f} MPa")


# --- Main Entry Point ---
if __name__ == '__main__':
    # You can modify CONFIG at the top of this file or pass arguments here
    
    # Check if all required files exist
    required_files = [
        CONFIG['model_path'],
        CONFIG['bert_path'],
        CONFIG['process_text_path'],
        CONFIG['composition_path']
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        print("ERROR: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all files are in place before running.")
        sys.exit(1)
    
    # Run optimization
    run_optimization(CONFIG)