"""Shared multimodal architecture used for yield-strength prediction.

This module preserves the model classes from the manuscript implementation.
PyTorch is an optional dependency and is imported only when this module is used.
"""

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None

    class _UnavailableModule:
        """Import-time stand-in used when optional ML dependencies are absent."""

    class _UnavailableNN:
        Module = _UnavailableModule

    nn = _UnavailableNN()


def _require_torch():
    if torch is None:
        raise RuntimeError("Install the agew-heabert 'ml' dependency group to use model classes")


ALL_ACT_LAYERS = (
    {
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
    }
    if torch is not None
    else {}
)


class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention mechanism for feature fusion."""

    def __init__(self, input_dim: int, num_heads: int = 8, dropout: float = 0.1):
        _require_torch()
        super().__init__()
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim),
            nn.Dropout(dropout),
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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        normalization: str = "layer_norm",
        activation: str = "relu",
        dropout_prob: float = 0.1,
    ):
        _require_torch()
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

        # Residual connection
        self.use_residual = in_features == out_features
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        pooling: str = "max",
        pool_kernel_size: int = 2,
    ):
        _require_torch()
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
        )
        self.act_fn = ALL_ACT_LAYERS[activation]()
        if pooling == "max":
            self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)
        else:
            self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size)

        # Residual connection
        self.use_residual = in_channels == out_channels
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str = "relu",
        pooling: str = "max",
        pool_kernel_size: int = 2,
    ):
        _require_torch()
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2),
        )
        self.act_fn = ALL_ACT_LAYERS[activation]()
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)
        else:
            self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size)

        # Residual connection
        self.use_residual = in_channels == out_channels
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
        self,
        simple_layer_list: list[int],
        concat_layer_list: list[int],
        seq_embed_con1d_list: list[int],
        seq_embed_fc_list: list[int],
        seq_embed_con2d_list: list[int],
        seq_embed_2d_fc_list: list[int],
        simple_layer_drop_prob: float = 0.0,
        concat_layer_drop_prob: float = 0.0,
        use_attention_fusion: bool = True,
        num_attention_heads: int = 8,
        num_chem_features: int = 7,
    ):
        _require_torch()
        super().__init__()

        self.use_attention_fusion = use_attention_fusion

        # Process embeddings
        self.simple_layer = self._create_mlp_pathway_with_residual(
            [768] + simple_layer_list, simple_layer_drop_prob
        )
        self.seq_embed_layer = self._create_1d_cnn_pathway_with_residual(
            1, seq_embed_con1d_list, seq_embed_fc_list, simple_layer_drop_prob
        )
        self.common_embed_layer = self._create_2d_cnn_pathway_with_residual(
            1, seq_embed_con2d_list, seq_embed_2d_fc_list, simple_layer_drop_prob
        )

        # Process chemical features
        if num_chem_features > 0:
            self.chem_feature_processor = nn.Sequential(
                nn.Linear(num_chem_features, 64),
                nn.ReLU(),
                nn.Dropout(simple_layer_drop_prob),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(simple_layer_drop_prob),
                nn.Linear(128, 64),
            )
        else:
            self.chem_feature_processor = None

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
                dropout=concat_layer_drop_prob,
            )

            # Final layers after attention
            self.final_layers = nn.Sequential(
                nn.LazyLinear(concat_layer_list[0]),
                nn.ReLU(),
                nn.Dropout(concat_layer_drop_prob),
                *[
                    nn.Sequential(
                        nn.Linear(concat_layer_list[i], concat_layer_list[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(concat_layer_drop_prob),
                    )
                    for i in range(len(concat_layer_list) - 1)
                ],
                nn.Linear(concat_layer_list[-1], 1),
            )
        else:
            # Original concatenation-based approach
            self.concat_layer = nn.Sequential(
                nn.LazyLinear(concat_layer_list[0]),
                nn.ReLU(),
                nn.Dropout(concat_layer_drop_prob),
                *[
                    nn.Sequential(
                        nn.Linear(concat_layer_list[i], concat_layer_list[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(concat_layer_drop_prob),
                    )
                    for i in range(len(concat_layer_list) - 1)
                ],
                nn.Linear(concat_layer_list[-1], 1),
            )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and not isinstance(module, nn.LazyLinear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _create_mlp_pathway_with_residual(self, layer_dims: list[int], dropout: float):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(
                FcUnitWithResidual(layer_dims[i], layer_dims[i + 1], dropout_prob=dropout)
            )
        return nn.Sequential(*layers)

    def _create_1d_cnn_pathway_with_residual(self, in_channels, conv1d_list, fc_list, dropout):
        layers = []
        current_channels = in_channels
        for out_channels in conv1d_list:
            layers.append(Cnn1dUnitWithResidual(current_channels, out_channels))
            current_channels = out_channels

        layers.extend([nn.Flatten(), nn.LazyLinear(fc_list[0])])
        for i in range(len(fc_list) - 1):
            layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_list[i], fc_list[i + 1])])
        return nn.Sequential(*layers)

    def _create_2d_cnn_pathway_with_residual(self, in_channels, conv2d_list, fc_list, dropout):
        layers = []
        current_channels = in_channels
        for out_channels in conv2d_list:
            layers.append(Cnn2dUnitWithResidual(current_channels, out_channels))
            current_channels = out_channels

        layers.extend([nn.Flatten(), nn.LazyLinear(fc_list[0])])
        for i in range(len(fc_list) - 1):
            layers.extend([nn.ReLU(), nn.Dropout(dropout), nn.Linear(fc_list[i], fc_list[i + 1])])
        return nn.Sequential(*layers)

    def forward(self, eles, chem_features, text_embeds, com_embeds, labels=None):
        # Process composition and text embeddings through their MLP pathways
        com_dense = self.simple_layer(com_embeds)
        text_dense = self.simple_layer(text_embeds)

        # Process the MLP outputs through their 1D CNN pathways
        com_1d_out = self.seq_embed_layer(com_dense.unsqueeze(1))
        text_1d_out = self.seq_embed_layer(text_dense.unsqueeze(1))

        # Create the cross-modal input by stacking the raw embeddings
        cross_modal_in = torch.stack([com_embeds, text_embeds], dim=1).unsqueeze(1)
        cross_modal_2d_out = self.common_embed_layer(cross_modal_in)

        # Process chemical features if available
        if self.chem_feature_processor is not None and chem_features.shape[1] > 0:
            chem_processed = self.chem_feature_processor(chem_features)
        else:
            chem_processed = None

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

            stacked_features = torch.cat(features, dim=1)  # [batch, num_modalities, dim]

            # Apply attention-based fusion
            fused_features = self.attention_fusion(stacked_features)

            # Flatten and pass through final layers
            fused_flat = fused_features.view(fused_features.size(0), -1)
            return self.final_layers(fused_flat)
        else:
            # Original concatenation approach
            if chem_processed is not None:
                combined = torch.cat(
                    [cross_modal_2d_out, com_1d_out, text_1d_out, chem_processed], dim=1
                )
            else:
                combined = torch.cat([cross_modal_2d_out, com_1d_out, text_1d_out], dim=1)
            return self.concat_layer(combined)
