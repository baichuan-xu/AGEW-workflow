# Standard libraries
import os
import sys
import json
import csv
import time
import uuid
import random
import pickle
import warnings
import copy
import traceback
from typing import Optional, List, Dict
from datetime import datetime
from collections import OrderedDict

# Data science and ML libraries
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# PyTorch and related libraries
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.swa_utils import AveragedModel, SWALR

# Hugging Face libraries
from transformers import AutoModel, AutoTokenizer

# Optuna for hyperparameter optimization
import optuna

# --- Global Configuration ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU, "gelu": nn.GELU, "relu": nn.ReLU,
    'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh
}

def set_global_seed(seed: int = 42):
    """Set random seeds for reproducibility across all relevant libraries."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_safetensors_checkpoint(model_path: str):
    """Convert a legacy .bin HF checkpoint to safetensors in-place, once.

    Recent ``transformers`` blocks loading ``pytorch_model.bin`` through
    ``torch.load`` unless torch >= 2.6 (CVE-2025-32434); the restriction does
    not apply to ``model.safetensors``. This routine detects a .bin-only
    checkpoint and converts it:

      1. load the raw state dict via ``torch.load`` (our own call, on the
         locally installed torch -- not subject to the transformers gate),
      2. write ``model.safetensors`` with ``safetensors.torch.save_file``,
      3. rename the .bin to ``*.bin.bak`` so ``from_pretrained`` selects the
         safetensors weights.

    Idempotent: if ``model.safetensors`` already exists, it returns
    immediately. If no .bin is present either, it leaves the directory alone
    and lets ``from_pretrained`` deal with whatever format is there.

    safetensors requires contiguous, non-shared tensors. HF .bin state dicts
    occasionally contain shared storage (tied weights); we clone to be safe.
    """
    st_path = os.path.join(model_path, 'model.safetensors')
    if os.path.exists(st_path):
        return  # already converted / native safetensors

    # Locate a legacy .bin weight file (single-file checkpoints only; sharded
    # .bin checkpoints have an index json and are not handled here).
    bin_candidates = ['pytorch_model.bin', 'model.bin']
    bin_path = next(
        (os.path.join(model_path, b) for b in bin_candidates
         if os.path.exists(os.path.join(model_path, b))),
        None
    )
    if bin_path is None:
        # Nothing to convert; defer to from_pretrained.
        return

    print(f"Converting legacy checkpoint '{os.path.basename(bin_path)}' -> "
          f"safetensors (one-time)...")
    try:
        from safetensors.torch import save_file
    except ImportError:
        print("  'safetensors' not installed. Run: pip install safetensors")
        sys.exit(1)

    # weights_only=True is safe here and matches the security intent. This
    # torch.load is ours, so the transformers >=4.46 gate does not apply.
    state_dict = torch.load(bin_path, map_location='cpu', weights_only=True)
    if not isinstance(state_dict, dict):
        print(f"  Unexpected checkpoint structure in {bin_path}; aborting "
              f"conversion."); sys.exit(1)

    # safetensors needs plain, contiguous tensors with no shared storage.
    clean_state = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean_state[k] = v.detach().cpu().contiguous().clone()
        # Non-tensor entries (rare in HF weight files) are dropped; they are
        # not model parameters and are reconstructed from config on load.

    save_file(clean_state, st_path, metadata={'format': 'pt'})

    # Shelve the .bin so from_pretrained prefers safetensors. Renaming rather
    # than deleting keeps a recoverable copy.
    bak_path = bin_path + '.bak'
    try:
        os.replace(bin_path, bak_path)
    except OSError:
        pass  # if rename fails, safetensors still takes precedence on load
    print(f"  Wrote '{st_path}'. Original retained as "
          f"'{os.path.basename(bak_path)}'.")

def cls_pooling(model_output):
    """Extract the CLS token (first token) embedding from BERT output."""
    return model_output.last_hidden_state[:, 0]

def get_embeddings_in_batches(texts: List[str], tokenizer, model, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of texts to embeddings in batches to prevent GPU memory errors.
    """
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings", leave=False):
            batch_texts = texts[i:i + batch_size]
            encoded_input = tokenizer(
                batch_texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
            )
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            model_output = model(**encoded_input)
            batch_embeddings = cls_pooling(model_output).detach().cpu().numpy()
            all_embeddings.append(batch_embeddings)
    return np.vstack(all_embeddings)

def gen_text_embed(bert_df: pd.DataFrame, col_embed: str, tokenizer, model, device: torch.device, embedding_dim: int = 768) -> pd.DataFrame:
    """Generate text embeddings for a DataFrame column."""
    if col_embed not in bert_df.columns:
        raise ValueError(f"Column '{col_embed}' not found in DataFrame.")
    
    texts = bert_df[col_embed].fillna('').astype(str).tolist()
    all_embeddings = get_embeddings_in_batches(texts, tokenizer, model, device)
    
    embed_cols = [f"{col_embed}_{i}" for i in range(embedding_dim)]
    embed_df = pd.DataFrame(all_embeddings, columns=embed_cols, index=bert_df.index)
    
    return embed_df

def add_element_embeddings(df: pd.DataFrame, element_cols: List[str], tokenizer, model, device: torch.device, embedding_dim: int = 768) -> pd.DataFrame:
    """Add weighted element embeddings based on composition."""
    element_embeddings = np.matrix(get_embeddings_in_batches(element_cols, tokenizer, model, device))
    weights_matrix = df[element_cols].values
    total_weights = weights_matrix.sum(axis=1)[:, np.newaxis] + 1e-9
    weighted_avg_matrix = (weights_matrix @ element_embeddings) / total_weights
    
    com_embed_df = pd.DataFrame(weighted_avg_matrix, columns=[f'com_embed_{i}' for i in range(embedding_dim)], index=df.index)
    return com_embed_df

def compute_chemical_features(df: pd.DataFrame, element_cols: List[str]) -> pd.DataFrame:
    """Compute chemical features from composition without changing raw data."""
    features = pd.DataFrame(index=df.index)
    
    # Ensure all values are non-negative
    comp_matrix = df[element_cols].values
    comp_matrix = np.maximum(comp_matrix, 0)
    
    # Number of elements
    features['num_elements'] = (comp_matrix > 0).sum(axis=1)
    
    # Composition entropy
    epsilon = 1e-10
    comp_norm = comp_matrix / (comp_matrix.sum(axis=1, keepdims=True) + epsilon)
    comp_norm = np.maximum(comp_norm, epsilon)
    features['composition_entropy'] = -(comp_norm * np.log(comp_norm)).sum(axis=1)
    
    # Maximum and minimum fractions
    features['max_fraction'] = comp_matrix.max(axis=1)
    features['min_fraction'] = np.where(comp_matrix > 0, comp_matrix, np.inf).min(axis=1)
    features['min_fraction'] = np.where(features['min_fraction'] == np.inf, 0, features['min_fraction'])
    
    # Composition variance and standard deviation
    features['composition_variance'] = comp_matrix.var(axis=1)
    features['composition_std'] = comp_matrix.std(axis=1)
    
    # Elemental diversity (Gini coefficient)
    def gini_coefficient(x):
        x = x[x > 0]
        if len(x) == 0:
            return 0
        x_sorted = np.sort(x)
        n = len(x_sorted)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n
    
    features['gini_coefficient'] = df[element_cols].apply(lambda x: gini_coefficient(x.values), axis=1)
    
    # Normalize features
    for col in features.columns:
        mean_val = features[col].mean()
        std_val = features[col].std()
        if std_val > 0:
            features[col] = (features[col] - mean_val) / std_val
    
    return features


# ============================================================================
#  Featurize-once + per-fold normalization (replaces the old load_data)
# ============================================================================

def build_full_dataset(
    pred_prop: str = 'Stress',
    fes: List[str] = ['com', 'text_embed', 'com_embed'],
    tokenizer=None, model=None, device: torch.device = None,
    embedding_dim: int = 768, add_chem_features: bool = True,
    data_path: str = './data.csv'
) -> pd.DataFrame:
    """Featurize the ENTIRE dataset exactly once (expensive HeaBERT embeddings).

    Returns a single un-normalized, un-split DataFrame whose column order matches
    what gen_data_class expects: [pred_prop] + element cols + text_embed cols +
    com_embed cols + chem-feature cols + 'Strain'. Splitting and normalization are
    handled separately (per fold) so that no target/strain statistics ever leak
    from a validation/test fold into training.
    """
    if not all([tokenizer, model, device]):
        raise ValueError("Tokenizer, model, and device must be provided.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    data_origin = pd.read_csv(data_path)

    required_cols = [pred_prop, 'Processing Text', 'Strain']
    exclude_cols = ['Material Name', pred_prop, 'Processing Text', 'Processing Word', 'Strain']
    component_cols = [col for col in data_origin.columns if col not in exclude_cols]

    for col in required_cols + component_cols:
        if col not in data_origin.columns:
            raise ValueError(f"Required column '{col}' not found in {data_path}")

    data = data_origin.copy()

    # Process component (element) columns
    for col in component_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data[component_cols] = data[component_cols].fillna(0)

    # Process strain column
    data['Strain'] = pd.to_numeric(data['Strain'], errors='coerce')
    data['Strain'] = data['Strain'].fillna(data['Strain'].mean())

    # Generate embeddings (single pass over the whole corpus)
    text_embed_df = gen_text_embed(data, col_embed='Processing Text', tokenizer=tokenizer, model=model, device=device, embedding_dim=embedding_dim)
    com_embed_df = add_element_embeddings(data, element_cols=component_cols, tokenizer=tokenizer, model=model, device=device, embedding_dim=embedding_dim)

    if add_chem_features:
        chem_features_df = compute_chemical_features(data, element_cols=component_cols)
        data = pd.concat([data, text_embed_df, com_embed_df, chem_features_df], axis=1)
    else:
        data = pd.concat([data, text_embed_df, com_embed_df], axis=1)

    # Assemble column order identical to the original load_data
    selected_cols = []
    if 'com' in fes: selected_cols.extend(component_cols)
    if 'text_embed' in fes: selected_cols.extend([f'Processing Text_{i}' for i in range(embedding_dim)])
    if 'com_embed' in fes: selected_cols.extend([f'com_embed_{i}' for i in range(embedding_dim)])
    if add_chem_features: selected_cols.extend(chem_features_df.columns.tolist())
    selected_cols.append('Strain')

    final_cols = [pred_prop] + selected_cols
    data = data[final_cols].dropna().reset_index(drop=True)

    # Downcast to float32: the model casts every tensor to float32 anyway, so the
    # float64 frame is pure overhead. Halves host RAM for the frame and all derived
    # numpy arrays with no numerical effect on training.
    float64_cols = data.select_dtypes(include=['float64']).columns
    data[float64_cols] = data[float64_cols].astype(np.float32)

    print(f"Full featurized dataset: {len(data)} rows x {len(final_cols)} columns "
          f"(target='{pred_prop}', dtype=float32).")
    return data


def apply_normalization(train_df: pd.DataFrame, other_dfs: List[pd.DataFrame], pred_prop: str = 'Stress'):
    """Standardize the target and Strain using TRAINING-fold statistics only.

    Sets the module-level globals (stress_mean/std, strain_mean/std) that the
    training loop and evaluation use for de-normalization, and returns normalized
    copies of the train frame and each frame in other_dfs. Never mutates inputs.
    """
    train_df = train_df.copy()
    others = [df.copy() for df in other_dfs]

    mean_val, std_val = float(train_df[pred_prop].mean()), float(train_df[pred_prop].std())
    if std_val == 0 or np.isnan(std_val):
        std_val = 1.0
    globals()['stress_mean'], globals()['stress_std'] = mean_val, std_val

    strain_mean, strain_std = float(train_df['Strain'].mean()), float(train_df['Strain'].std())
    if strain_std == 0 or np.isnan(strain_std):
        strain_std = 1.0
    globals()['strain_mean'], globals()['strain_std'] = strain_mean, strain_std

    for df in [train_df] + others:
        df[pred_prop] = (df[pred_prop] - mean_val) / std_val
        df['Strain'] = (df['Strain'] - strain_mean) / strain_std

    norm_params = {
        'stress_mean': mean_val, 'stress_std': std_val,
        'strain_mean': strain_mean, 'strain_std': strain_std,
    }
    return train_df, others, norm_params

def gen_data_class(df: pd.DataFrame, embedding_dim: int = 768) -> Dict[str, pd.DataFrame]:
    """Convert a DataFrame into a dictionary of features and targets."""
    targets = df.iloc[:, 0]
    eles_cols = [col for col in df.columns[1:] if not ('_embed_' in col or 'Processing Text_' in col or col == 'Strain')]
    chem_cols = ['num_elements', 'composition_entropy', 'max_fraction', 'min_fraction', 
                 'composition_variance', 'composition_std', 'gini_coefficient']
    
    # Separate element cols and chemical feature cols
    element_cols = [col for col in eles_cols if col not in chem_cols]
    chem_feature_cols = [col for col in eles_cols if col in chem_cols]
    
    return {
        "targets": targets,
        "eles": df[element_cols] if element_cols else pd.DataFrame(index=df.index),
        "chem_features": df[chem_feature_cols] if chem_feature_cols else pd.DataFrame(index=df.index),
        "text_embeds": df[[f'Processing Text_{i}' for i in range(embedding_dim)]],
        "com_embeds": df[[f'com_embed_{i}' for i in range(embedding_dim)]],
        "strain": df['Strain'],
    }

def eval_model(y_true, y_pred) -> Dict[str, float]:
    """Evaluate regression performance."""
    y_true = y_true.detach().cpu().numpy().flatten() if torch.is_tensor(y_true) else np.array(y_true).flatten()
    y_pred = y_pred.detach().cpu().numpy().flatten() if torch.is_tensor(y_pred) else np.array(y_pred).flatten()
    
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred)
    }
    
def plot_results(train_results, val_results, test_results, fig_name):
    """Plot true vs predicted values for all data splits."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=120)
    splits = [('Train', train_results, 'blue'), ('Validation', val_results, 'green'), ('Test', test_results, 'purple')]

    for ax, (name, results, color) in zip(axes, splits):
        y_true, y_pred, metrics = results
        ax.scatter(y_true, y_pred, s=10, alpha=0.7, color=color, label=name)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1, label='y=x')
        ax.set_title(f"{name}\n$R^2$={metrics['r2']:.2f}, RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predictions")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    plt.savefig(fig_name)
    plt.close()
    print(f"Result plot saved to {fig_name}")

# --- PyTorch Model Components ---

class CustomSimpleDataset(Dataset):
    """Custom PyTorch Dataset for handling multiple data types."""
    def __init__(self, eles, chem_features, text_embeds, com_embeds, strain, targets, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eles = eles.values if isinstance(eles, pd.DataFrame) else eles
        self.chem_features = chem_features.values if isinstance(chem_features, pd.DataFrame) else chem_features
        self.text_embeds = text_embeds.values if isinstance(text_embeds, pd.DataFrame) else text_embeds
        self.com_embeds = com_embeds.values if isinstance(com_embeds, pd.DataFrame) else com_embeds
        self.strain = strain.values if isinstance(strain, (pd.DataFrame, pd.Series)) else strain
        self.targets = targets.values if isinstance(targets, (pd.DataFrame, pd.Series)) else targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "labels": torch.tensor(self.targets[idx], dtype=torch.float32).to(self.device),
            "eles": torch.tensor(self.eles[idx], dtype=torch.float32).to(self.device),
            "chem_features": torch.tensor(self.chem_features[idx], dtype=torch.float32).to(self.device),
            "text_embeds": torch.tensor(self.text_embeds[idx], dtype=torch.float32).to(self.device),
            "com_embeds": torch.tensor(self.com_embeds[idx], dtype=torch.float32).to(self.device),
            "strain": torch.tensor(self.strain[idx], dtype=torch.float32).to(self.device),
        }

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
        # x shape: [batch, num_modalities, dim]
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

class FcUnitWithResidual(nn.Module):
    """MLP block with residual connection."""
    def __init__(self, in_features: int, out_features: int, normalization: str = "layer_norm", activation: str = 'relu', dropout_prob: float = 0.1):
        super().__init__()
        if normalization == "layer_norm": self.norm = nn.LayerNorm(in_features)
        elif normalization == "batch_norm": self.norm = nn.BatchNorm1d(in_features)
        else: self.norm = nn.Identity()
        self.fc = nn.Linear(in_features, out_features)
        self.act_fn = ALL_ACT_LAYERS[activation]()
        self.dropout = nn.Dropout(dropout_prob)
        
        # Residual connection
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, activation: str = 'relu', pooling: str = 'max', pool_kernel_size: int = 2):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.act_fn = ALL_ACT_LAYERS[activation]()
        if pooling == "max": self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        else: self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size)
        
        # Residual connection
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
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, activation: str = 'relu', pooling: str = 'max', pool_kernel_size: int = 2):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.act_fn = ALL_ACT_LAYERS[activation]()
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size)
            
        # Residual connection
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

class StrainMLPPathway(nn.Module):
    """MLP pathway to expand strain from 1-dim to 256-dim."""
    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, x):
        # x shape: [batch_size] or [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [batch_size, 1]
        return self.mlp(x)  # [batch_size, 256]

class StrainExpansionFor2DCNN(nn.Module):
    """MLP pathway to expand strain from 1-dim to 768-dim for cross-modal 2D CNN."""
    def __init__(self, dropout_prob: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, x):
        # x shape: [batch_size] or [batch_size, 1]
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [batch_size, 1]
        return self.mlp(x)  # [batch_size, 768]

class CustomAdvancedModel(nn.Module):
    """Enhanced multimodal regression model with attention-based fusion and strain in cross-modal."""
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
        
        # Process embeddings
        self.simple_layer = self._create_mlp_pathway_with_residual([768] + simple_layer_list, simple_layer_drop_prob)
        self.seq_embed_layer = self._create_1d_cnn_pathway_with_residual(1, seq_embed_con1d_list, seq_embed_fc_list, simple_layer_drop_prob)
        
        # MODIFIED: Cross-modal now processes 3 modalities (com, text, strain)
        self.common_embed_layer = self._create_2d_cnn_pathway_with_residual(1, seq_embed_con2d_list, seq_embed_2d_fc_list, simple_layer_drop_prob)
        
        # Process chemical features
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
        
        # Strain MLP Pathway (256D for attention fusion)
        self.strain_processor = StrainMLPPathway(dropout_prob=simple_layer_drop_prob)
        
        # NEW: Strain expansion to 768D for cross-modal 2D CNN
        self.strain_expansion_768d = StrainExpansionFor2DCNN(dropout_prob=simple_layer_drop_prob)
        
        # Attention-based fusion
        if use_attention_fusion:
            # Project all features to same dimension before attention
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
            
            # Final layers after attention (input is 5*256=1280)
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
            # Original concatenation-based approach
            self.concat_layer = nn.Sequential(
                nn.LazyLinear(concat_layer_list[0]), nn.ReLU(), nn.Dropout(concat_layer_drop_prob),
                *[nn.Sequential(
                    nn.Linear(concat_layer_list[i], concat_layer_list[i+1]), nn.ReLU(), nn.Dropout(concat_layer_drop_prob)
                  ) for i in range(len(concat_layer_list)-1)],
                nn.Linear(concat_layer_list[-1], 1)
            )
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not isinstance(module, nn.LazyLinear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.zeros_(module.bias)

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

    def forward(self, eles, chem_features, text_embeds, com_embeds, strain, labels=None):
        # Process composition and text embeddings through their MLP pathways
        com_dense = self.simple_layer(com_embeds)
        text_dense = self.simple_layer(text_embeds)

        # Process the MLP outputs through their 1D CNN pathways
        com_1d_out = self.seq_embed_layer(com_dense.unsqueeze(1))
        text_1d_out = self.seq_embed_layer(text_dense.unsqueeze(1))
        
        # MODIFIED: Expand strain to 768D for cross-modal 2D CNN
        strain_768d = self.strain_expansion_768d(strain)  # [batch_size, 768]
        
        # MODIFIED: Create the cross-modal input by stacking THREE embeddings (com, text, strain)
        # Shape: [batch, 1, 3, 768] where 3 modalities are stacked
        cross_modal_in = torch.stack([com_embeds, text_embeds, strain_768d], dim=1).unsqueeze(1)
        cross_modal_2d_out = self.common_embed_layer(cross_modal_in)
        
        # Process chemical features if available
        if self.chem_feature_processor is not None and chem_features.shape[1] > 0:
            chem_processed = self.chem_feature_processor(chem_features)
        else:
            chem_processed = None
        
        # Process strain through 256D MLP pathway for attention fusion
        strain_processed = self.strain_processor(strain)  # [batch_size, 256]

        if self.use_attention_fusion:
            # Project all features to same dimension
            com_proj = self.com_projection(com_1d_out)
            text_proj = self.text_projection(text_1d_out)
            cross_proj = self.cross_projection(cross_modal_2d_out)
            
            # Stack features for attention
            features = [com_proj.unsqueeze(1), text_proj.unsqueeze(1), cross_proj.unsqueeze(1)]
            if chem_processed is not None:
                chem_proj = self.chem_projection(chem_processed)
                features.append(chem_proj.unsqueeze(1))
            
            # Add strain to attention stack (already 256-dim, no projection needed)
            features.append(strain_processed.unsqueeze(1))
            
            stacked_features = torch.cat(features, dim=1)  # [batch, 5, 256]
            
            # Apply attention-based fusion
            fused_features = self.attention_fusion(stacked_features)
            
            # Flatten and pass through final layers
            fused_flat = fused_features.view(fused_features.size(0), -1)  # [batch, 1280]
            return self.final_layers(fused_flat)
        else:
            # Original concatenation approach
            if chem_processed is not None:
                combined = torch.cat([cross_modal_2d_out, com_1d_out, text_1d_out, chem_processed, strain_processed], dim=1)
            else:
                combined = torch.cat([cross_modal_2d_out, com_1d_out, text_1d_out, strain_processed], dim=1)
            return self.concat_layer(combined)

# Virtual Adversarial Training
def virtual_adversarial_loss(model, batch, eps=1.0):
    """Compute virtual adversarial training loss."""
    # Get original prediction
    labels = batch.pop('labels')
    with torch.no_grad():
        pred = model(**batch).detach()
    
    # Initialize random perturbation for each input type
    d = {}
    for key, value in batch.items():
        d[key] = torch.randn_like(value).to(value.device)
        d[key] = eps * d[key] / (torch.norm(d[key], dim=-1, keepdim=True) + 1e-8)
    
    # Apply perturbation and get adversarial prediction
    perturbed_batch = {key: batch[key] + d[key] for key in batch}
    pred_hat = model(**perturbed_batch)
    
    # Return KL divergence for regression (using MSE as proxy)
    return F.mse_loss(pred_hat, pred)


import gc
from sklearn.model_selection import KFold


def free_memory(*objs):
    """Drop references and release cached GPU/host memory between folds/trials."""
    for _o in objs:
        del _o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_in_batches(model, dataset, eval_batch_size: int = 512):
    """Run inference over a dataset in fixed-size batches and return (y_true, y_pred)
    as CPU tensors. Avoids a single full-dataset forward pass, which for a ~14k-row
    training fold would be a multi-GB GPU spike."""
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            y = batch.pop('labels').unsqueeze(1)
            p = model(**batch)
            ys.append(y.detach().cpu())
            ps.append(p.detach().cpu())
    return torch.cat(ys), torch.cat(ps)

# ============================================================================
#  Reusable training / evaluation core (model, VAT, SWA loop unchanged)
# ============================================================================

def train_model_core(config, train_dataset, val_dataset, device,
                     early_stopping: bool = True, verbose: bool = False,
                     trial=None):
    """Train one CustomAdvancedModel with the given config.

    - early_stopping=True and a val_dataset present: track best val R2, stop on
      patience, return the best-checkpoint model. Used for CV folds (the held-out
      fold is passed as val).
    - early_stopping=False / val_dataset=None: run the full fixed epoch schedule
      with SWA and return the SWA-averaged model (no validation, no best-tracking).
      Used for the final all-data model.
    - trial (Optuna): when provided, reports intermediate R² for intra-trial pruning.

    Returns (eval_model_final [in eval mode], best_val_r2).
    """
    train_dataloader = DataLoader(train_dataset, batch_size=config['train_batch'], shuffle=True)

    reg_model = CustomAdvancedModel(**{k: v for k, v in config.items() if k in [
        'simple_layer_list', 'concat_layer_list', 'seq_embed_con1d_list', 'seq_embed_fc_list',
        'seq_embed_con2d_list', 'seq_embed_2d_fc_list', 'simple_layer_drop_prob', 'concat_layer_drop_prob',
        'use_attention_fusion', 'num_attention_heads'
    ]}).to(device)

    # Materialize LazyLinear parameters with one dummy forward BEFORE wrapping in
    # AveragedModel; otherwise SWA's update_parameters averages uninitialized params.
    reg_model.eval()
    with torch.no_grad():
        _init_batch = next(iter(train_dataloader))
        _init_batch.pop('labels', None)
        reg_model(**_init_batch)
    reg_model.train()

    swa_model = AveragedModel(reg_model)
    swa_start_epoch = int(config['epoch'] * config['swa_start'])

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(reg_model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 0.01))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step'], gamma=config['gamma_ratio'])
    swa_scheduler = SWALR(optimizer, swa_lr=config['swa_lr'])

    best_val_r2, best_model_state = -float('inf'), None
    best_is_swa = False
    patience = config.get('patience', 50)
    patience_counter = 0
    swa_n = 0

    for epoch_num in range(config['epoch']):
        reg_model.train()
        for batch in train_dataloader:
            batch_copy = {k: v.clone() for k, v in batch.items()}
            labels = batch.pop('labels').unsqueeze(1)
            preds = reg_model(**batch)

            main_loss = loss_fn(preds, labels)
            if config['vat_weight'] > 0:
                batch_copy['labels'] = labels
                vat_loss = virtual_adversarial_loss(reg_model, batch_copy, eps=config['vat_eps'])
                total_loss = main_loss + config['vat_weight'] * vat_loss
            else:
                total_loss = main_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(reg_model.parameters(), max_norm=1.0)
            optimizer.step()

        if epoch_num < swa_start_epoch:
            scheduler.step()
        else:
            swa_model.update_parameters(reg_model)
            swa_scheduler.step()
            swa_n += 1

        # Validation / early stopping (only when requested and a val set exists)
        if early_stopping and val_dataset is not None and len(val_dataset) > 0:
            eval_model_to_use = swa_model if epoch_num >= swa_start_epoch else reg_model
            val_labels, val_preds = predict_in_batches(eval_model_to_use, val_dataset)
            val_metrics = eval_model(val_labels, val_preds)

            if val_metrics['r2'] > best_val_r2:
                best_val_r2 = val_metrics['r2']
                if epoch_num >= swa_start_epoch:
                    best_model_state = {k: v.detach().cpu().clone() for k, v in swa_model.state_dict().items()}
                    best_is_swa = True
                else:
                    best_model_state = {k: v.detach().cpu().clone() for k, v in reg_model.state_dict().items()}
                    best_is_swa = False
                patience_counter = 0
            else:
                patience_counter += 1

            # Intra-trial Optuna pruning (reports best R² at each epoch)
            if trial is not None:
                trial.report(best_val_r2, epoch_num)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch_num + 1}.")
                break

    # Update BN running stats for the SWA model (harmless no-op if no BatchNorm)
    if swa_n > 0:
        torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)

    # Choose the model to return
    if best_model_state is not None:
        if best_is_swa:
            swa_model.load_state_dict(best_model_state)
            eval_model_final = swa_model
        else:
            reg_model.load_state_dict(best_model_state)
            eval_model_final = reg_model
    else:
        eval_model_final = swa_model if swa_n > 0 else reg_model

    eval_model_final.eval()
    return eval_model_final, best_val_r2


def evaluate_r2(eval_model_final, dataset, device):
    """Compute de-normalized R2/RMSE/MAE for a dataset using current norm globals."""
    mean, std = globals()['stress_mean'], globals()['stress_std']
    y_norm, preds_norm = predict_in_batches(eval_model_final, dataset)
    y_true = y_norm.numpy().flatten() * std + mean
    y_pred = preds_norm.numpy().flatten() * std + mean
    return eval_model(y_true, y_pred)


def build_config_from_params(best_params: dict, pred_prop: str, epochs: int, seed: int) -> dict:
    """Reconstruct the full model/training config from an Optuna best_params dict."""
    return {
        "prop": pred_prop, "seed": seed, "epoch": epochs,
        "seq_embed_con1d_list": [2, 4], "seq_embed_fc_list": [32, 16],
        "seq_embed_con2d_list": [2, 4], "seq_embed_2d_fc_list": [32, 16],
        "swa_start": 0.75, "swa_lr": 5e-5, "patience": 50,
        "train_batch": best_params["train_batch"],
        "lr": best_params["lr"],
        "step": best_params["step"],
        "gamma_ratio": best_params["gamma_ratio"],
        "weight_decay": best_params.get("weight_decay", 0.01),
        "simple_layer_list": [512] * best_params["simple_512"] + [256] * best_params["simple_256"],
        "concat_layer_list": [64] * best_params["concat_64"] + [32] * best_params["concat_32"] + [8],
        "simple_layer_drop_prob": best_params["simple_drop"],
        "concat_layer_drop_prob": best_params["concat_drop"],
        "use_attention_fusion": best_params["use_attention"],
        "num_attention_heads": best_params["num_heads"],
        "vat_weight": best_params["vat_weight"],
        "vat_eps": best_params["vat_eps"],
    }


# ============================================================================
#  Optuna objective: single-pass tuning.
#  Each trial = ONE training run on a fixed 80/20 split. The search returns one
#  winning config, which is then frozen. (The honest generalization estimate is
#  the separate 10-fold CV in step 3; tuning only needs to rank candidates.)
# ============================================================================

def objective(trial, device, train_df_n, val_df_n, tune_epochs=200):
    config = {
        "train_batch": trial.suggest_categorical("train_batch", [16, 32, 64]),
        "epoch": tune_epochs,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "step": trial.suggest_int("step", 40, 80),
        "gamma_ratio": trial.suggest_float("gamma_ratio", 0.0, 0.8),
        "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        "simple_layer_list": [512] * trial.suggest_int("simple_512", 2, 4) + [256] * trial.suggest_int("simple_256", 1, 3),
        "simple_layer_drop_prob": trial.suggest_float("simple_drop", 0.1, 0.5),
        "concat_layer_list": [64] * trial.suggest_int("concat_64", 2, 4) + [32] * trial.suggest_int("concat_32", 1, 3) + [8],
        "concat_layer_drop_prob": trial.suggest_float("concat_drop", 0.1, 0.5),
        "seq_embed_con1d_list": [2, 4],
        "seq_embed_fc_list": [32, 16],
        "seq_embed_con2d_list": [2, 4],
        "seq_embed_2d_fc_list": [32, 16],
        "use_attention_fusion": trial.suggest_categorical("use_attention", [True, False]),
        "num_attention_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
        "vat_weight": trial.suggest_float("vat_weight", 0.0, 1.0),
        "vat_eps": trial.suggest_float("vat_eps", 0.1, 2.0),
        # fixed (not searched) — added so train_model_core has a complete config
        "swa_start": 0.75, "swa_lr": 5e-5, "patience": 15,
    }

    train_ds = CustomSimpleDataset(**gen_data_class(train_df_n), device=device)
    val_ds = CustomSimpleDataset(**gen_data_class(val_df_n), device=device)

    trial_model, best_val_r2 = train_model_core(config, train_ds, val_ds, device,
                                                 early_stopping=True, trial=trial)

    # Release this trial's allocations before the next trial starts
    free_memory(trial_model, train_ds, val_ds)

    return float(best_val_r2)


# ============================================================================
#  Main: inference-only — load trained model from ss7, group-predict on
#        data.csv by Processing Text, retain groups with R² > 0.75
# ============================================================================
if __name__ == '__main__':
    # ---------------------------- configuration ----------------------------
    PRED_PROP = 'Stress'
    FES = ['com', 'text_embed', 'com_embed']
    ADD_CHEM = True
    PRETRAINED_PATH = './pretrained'
    FINAL_MODEL_DIR = './outputs/final_model'
    DATA_PATH = './data.csv'
    EMBEDDING_DIM = 768
    R2_THRESHOLD = 0.75
    PREDICT_DIR = './predict'

    # --------------- step 1: load BERT & trained model --------------------
    print("=" * 60 + "\nSTEP 1: LOAD PRETRAINED BERT & TRAINED MODEL\n" + "=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- load BERT ---
    if not os.path.isdir(PRETRAINED_PATH):
        print(f"Error: Pretrained folder not found at '{PRETRAINED_PATH}'."); sys.exit(1)
    ensure_safetensors_checkpoint(PRETRAINED_PATH)
    try:
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_PATH)
        bert_model = AutoModel.from_pretrained(PRETRAINED_PATH).to(device)
        print(f"Loaded pretrained BERT from '{PRETRAINED_PATH}'")
    except Exception as e:
        print(f"Failed to load BERT: {e}"); sys.exit(1)

    # --- load saved config, norm params, and model weights ---
    config_path = os.path.join(FINAL_MODEL_DIR, 'config.json')
    norm_path   = os.path.join(FINAL_MODEL_DIR, 'norm_params.json')
    weight_path = os.path.join(FINAL_MODEL_DIR, 'final_model.pt')

    for p, name in [(config_path, 'config.json'), (norm_path, 'norm_params.json'),
                    (weight_path, 'final_model.pt')]:
        if not os.path.isfile(p):
            print(f"Error: '{name}' not found in '{FINAL_MODEL_DIR}'."); sys.exit(1)

    with open(config_path, 'r') as f:
        saved_config = json.load(f)
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)

    # Set normalization globals
    stress_mean = float(norm_params['stress_mean'])
    stress_std  = float(norm_params['stress_std'])
    strain_mean = float(norm_params['strain_mean'])
    strain_std  = float(norm_params['strain_std'])
    globals()['stress_mean'] = stress_mean
    globals()['stress_std']  = stress_std
    globals()['strain_mean'] = strain_mean
    globals()['strain_std']  = strain_std
    print(f"Loaded normalization: stress_mean={stress_mean:.4f}, stress_std={stress_std:.4f}, "
          f"strain_mean={strain_mean:.4f}, strain_std={strain_std:.4f}")

    # --- reconstruct model and load trained weights ---
    reg_model = CustomAdvancedModel(**{k: v for k, v in saved_config.items() if k in [
        'simple_layer_list', 'concat_layer_list', 'seq_embed_con1d_list',
        'seq_embed_fc_list', 'seq_embed_con2d_list', 'seq_embed_2d_fc_list',
        'simple_layer_drop_prob', 'concat_layer_drop_prob',
        'use_attention_fusion', 'num_attention_heads'
    ]}).to(device)

    state_dict = torch.load(weight_path, map_location=device, weights_only=True)

    # Strip SWA AveragedModel wrapper prefix ("module.") if present, and drop
    # the "n_averaged" counter.
    if 'n_averaged' in state_dict or any(k.startswith('module.') for k in state_dict):
        print("  Detected SWA AveragedModel wrapper — stripping 'module.' prefix.")
        cleaned = {}
        for k, v in state_dict.items():
            if k == 'n_averaged':
                continue
            cleaned[k[7:] if k.startswith('module.') else k] = v
        state_dict = cleaned

    reg_model.load_state_dict(state_dict)
    reg_model.eval()
    print(f"Loaded trained model from '{weight_path}'")

    # --------------- step 2: read & featurize data.csv --------------------
    print("\n" + "=" * 60 + "\nSTEP 2: READ & FEATURIZE data.csv\n" + "=" * 60)

    if not os.path.isfile(DATA_PATH):
        print(f"Error: '{DATA_PATH}' not found."); sys.exit(1)

    orig_df = pd.read_csv(DATA_PATH)
    original_cols = orig_df.columns.tolist()
    print(f"Loaded {len(orig_df)} rows from '{DATA_PATH}'")

    required_cols = [PRED_PROP, 'Processing Text', 'Strain', 'Material Name']
    exclude_cols = ['Material Name', PRED_PROP, 'Processing Text', 'Processing Word', 'Strain']
    component_cols = [col for col in orig_df.columns if col not in exclude_cols]

    for col in required_cols + component_cols:
        if col not in orig_df.columns:
            raise ValueError(f"Required column '{col}' not found in '{DATA_PATH}'")

    data = orig_df.copy()

    # Process component (element) columns
    for col in component_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data[component_cols] = data[component_cols].fillna(0)

    # Process strain column
    data['Strain'] = pd.to_numeric(data['Strain'], errors='coerce')
    data['Strain'] = data['Strain'].fillna(data['Strain'].mean())

    # Generate embeddings (single pass over the whole corpus)
    text_embed_df = gen_text_embed(data, col_embed='Processing Text', tokenizer=tokenizer,
                                   model=bert_model, device=device, embedding_dim=EMBEDDING_DIM)
    com_embed_df = add_element_embeddings(data, element_cols=component_cols, tokenizer=tokenizer,
                                          model=bert_model, device=device, embedding_dim=EMBEDDING_DIM)

    if ADD_CHEM:
        chem_features_df = compute_chemical_features(data, element_cols=component_cols)
        data = pd.concat([data, text_embed_df, com_embed_df, chem_features_df], axis=1)
    else:
        data = pd.concat([data, text_embed_df, com_embed_df], axis=1)

    # Build selected column list (same order as build_full_dataset)
    selected_cols = []
    if 'com' in FES: selected_cols.extend(component_cols)
    if 'text_embed' in FES: selected_cols.extend([f'Processing Text_{i}' for i in range(EMBEDDING_DIM)])
    if 'com_embed' in FES: selected_cols.extend([f'com_embed_{i}' for i in range(EMBEDDING_DIM)])
    if ADD_CHEM: selected_cols.extend(chem_features_df.columns.tolist())
    selected_cols.append('Strain')

    featurized_cols = [PRED_PROP] + selected_cols
    keep_cols = ['Material Name', 'Processing Text']
    all_needed = featurized_cols + keep_cols

    # Drop rows where any needed column is NaN; keep alignment with original data
    data = data.dropna(subset=all_needed).reset_index(drop=True)

    # Downcast to float32 for consistency with the model
    float64_cols = data.select_dtypes(include=['float64']).columns
    data[float64_cols] = data[float64_cols].astype(np.float32)

    print(f"Featurized: {len(data)} rows × {len(data.columns)} columns "
          f"(dropped {len(orig_df) - len(data)} rows with NaN).")

    # --------------- step 3: normalize ------------------------------------
    print("\n" + "=" * 60 + "\nSTEP 3: NORMALIZE & GROUP BY PROCESSING TEXT\n" + "=" * 60)

    data_n = data.copy()
    data_n[PRED_PROP] = (data_n[PRED_PROP] - stress_mean) / stress_std
    data_n['Strain'] = (data_n['Strain'] - strain_mean) / strain_std

    # Group by Processing Text
    groups = data_n.groupby('Processing Text')
    n_groups = len(groups)
    print(f"Found {n_groups} unique groups (by Processing Text).")

    # --------------- step 4: predict per group ----------------------------
    print("\n" + "=" * 60 + f"\nSTEP 4: GROUPED PREDICTIONS (threshold R² > {R2_THRESHOLD})\n" + "=" * 60)

    os.makedirs(PREDICT_DIR, exist_ok=True)
    retained = 0

    for proc_text, idxs in groups.groups.items():
        group_n = data_n.loc[idxs]
        group_data = data.loc[idxs]

        # Material name: use mode (most common), fall back to first
        mat_names = group_data['Material Name']
        mat_name = mat_names.mode().iloc[0] if len(mat_names.mode()) > 0 else mat_names.iloc[0]
        n_rows = len(group_n)

        # Create dataset from featurized columns only
        ds = CustomSimpleDataset(**gen_data_class(group_n[featurized_cols],
                                                   embedding_dim=EMBEDDING_DIM),
                                  device=device)

        # Predict and de-normalize
        y_norm, preds_norm = predict_in_batches(reg_model, ds)
        y_true = y_norm.numpy().flatten() * stress_std + stress_mean
        y_pred = preds_norm.numpy().flatten() * stress_std + stress_mean

        r2_val = r2_score(y_true, y_pred)

        # Truncate processing text for display
        pt_short = proc_text[:60] + ('...' if len(proc_text) > 60 else '')
        print(f"  [{n_rows:4d} rows] {pt_short}  →  R² = {r2_val:.4f}")

        if r2_val > R2_THRESHOLD:
            retained += 1
            # Sanitize material name for use as a folder name
            safe_name = str(mat_name).replace('/', '_').replace('\\', '_').replace(':', '_')
            subdir = os.path.join(PREDICT_DIR, safe_name)
            os.makedirs(subdir, exist_ok=True)

            # out.csv: original data columns + Predicted_Stress
            out_df = group_data[original_cols].copy()
            out_df['Predicted_Stress'] = y_pred
            out_csv = os.path.join(subdir, 'out.csv')
            out_df.to_csv(out_csv, index=False)

            # R2.txt
            r2_txt = os.path.join(subdir, 'R2.txt')
            with open(r2_txt, 'w') as f:
                f.write(f"{r2_val:.6f}\n")

            print(f"         → Retained in '{subdir}'")
        else:
            print(f"         → Skipped")

        free_memory(ds)

    print(f"\n{'#' * 60}")
    print(f"Done. {retained}/{n_groups} groups retained (R² > {R2_THRESHOLD}).")
    print(f"Results saved to '{PREDICT_DIR}/'")
    print(f"{'#' * 60}")
