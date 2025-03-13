"""
Core functionality of the package.
Functions of the package are her. 
If you develop them in a notebook, mpve them here when ready.
"""

# attention model with residual connectoin
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Assuming you have lists/arrays of training metrics
def plot_training_curves(
    train_losses, 
    val_losses=None, 
    epochs=None,
    figsize=(10, 6)
):
    plt.figure(figsize=figsize)
    epochs = epochs or range(1, len(train_losses) + 1)
    
    # Plot training loss
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    # Plot validation loss if available
    if val_losses:
        plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding for the transformer model.

    Attributes:
        pe (Tensor): A buffer containing the positional encodings.

    Args:
        d_model (int): The dimension of the model.
        max_len (int, optional): The maximum length of the sequence. Default is 1000.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(-2)]

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism as described in the "Attention is All You Need" paper.

    Attributes:
        d_model (int): The dimensionality of the input and output features.
        n_heads (int): The number of attention heads.
        d_k (int): The dimensionality of each attention head (d_model divided by n_heads).
        W_q (nn.Linear): Linear layer to project the input query to the query space.
        W_k (nn.Linear): Linear layer to project the input key to the key space.
        W_v (nn.Linear): Linear layer to project the input value to the value space.
        W_o (nn.Linear): Linear layer to project the concatenated output of all attention heads.
        dropout (nn.Dropout): Dropout layer applied to the attention weights.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Initializes the MultiHeadAttention module.

        Raises:
        AssertionError: If d_model is not divisible by n_heads.

        Args:
            d_model (int): The dimensionality of the input and output features.
            n_heads (int): The number of attention heads.
            dropout (float, optional): Dropout probability for the attention weights. Default is 0.1.
        """
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Performs the forward pass of the Multi-Head Attention mechanism.

        Args:
            query (Tensor) Q: The input query tensor of shape (batch_size, seq_len, d_model).
            key (Tensor) K: The input key tensor of shape (batch_size, seq_len, d_model).
            value (Tensor) V: The input value tensor of shape (batch_size, seq_len, d_model).
            mask (Tensor, optional): The mask tensor to apply to the attention scores. Default is None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - output (Tensor): The output tensor of shape (batch_size, seq_len, d_model).
                - attention_weights (Tensor): The attention weights tensor of shape (batch_size, n_heads, seq_len, seq_len).
        """
        batch_size = query.size(0)
        seq_len = query.size(1)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output, attention_weights

class FeedForward(nn.Module):
    """
    Implements the feed-forward neural network used in the transformer model.
    
    Model architecture:
    - Linear layer with ReLU activation
    - Dropout layer
    - Linear layer
    """
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    
class ClassificationHead(nn.Module):
    """
    A modular classification head that handles feature pooling and classification.
    
    Model architecture:
    - Feature pooling (learned or mean)
    - Classifier MLP with:
        - Linear layer
        - ReLU activation
        - Dropout
        - Final linear projection
    """
    def __init__(self, d_model, n_features, n_classes, dropout=0.1, use_learned_pooling=True):
        super().__init__()
        
        # Feature pooling
        self.use_learned_pooling = use_learned_pooling
        if use_learned_pooling:
            self.feature_pool = nn.Linear(n_features, 1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, n_features, d_model]
        
        # Pool across features
        if self.use_learned_pooling:
            x = x.transpose(1, 2)  # [batch_size, d_model, n_features]
            x = self.feature_pool(x)  # [batch_size, d_model, 1]
            x = x.squeeze(-1)  # [batch_size, d_model]
        else:
            x = x.mean(dim=1)  # [batch_size, d_model]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, n_classes]
        return logits


class DualAttentionModule(nn.Module):
    """
    A module that implements dual attention mechanism (feature and time attention).
    
    Model architecture:
    - Input projection
    - Positional encoding   
    - Feature attention block
      - inlcudig feature feed-forward MLP
    - Time attention block
      - inlcudig time feed-forward MLP
    - Output layer
      - including final feed-forward MLP
    - Residual connections throughout:
        - Feature attention block
        - Time attention block
        - Each feed-forward MLP block
    """
    def __init__(self, n_features, time_steps, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Input projection (only needed for first module)
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Feature attention block
        self.feature_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_ff = FeedForward(d_model, dropout=dropout)
        self.feature_ff_norm = nn.LayerNorm(d_model)

        # Time attention block
        self.time_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_ff = FeedForward(d_model, dropout=dropout)
        self.time_ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_first_module=False):
        batch_size, n_features, time_steps, d_model = x.shape

        # Apply input projection and positional encoding only for first module
        if is_first_module:
            x = self.input_projection(x.unsqueeze(-1))  # Project to d_model
            x = self.pos_encoder(x)

        # Feature attention block
        x_feat = x.transpose(1, 2).reshape(batch_size * time_steps, n_features, self.d_model)
        identity = x_feat

        x_feat, feat_weights = self.feature_attention(x_feat, x_feat, x_feat)
        x_feat = identity + self.dropout(x_feat)
        x_feat = self.feature_norm(x_feat)

        identity = x_feat
        x_feat = identity + self.dropout(self.feature_ff(x_feat))
        x_feat = self.feature_ff_norm(x_feat)

        x = x_feat.view(batch_size, time_steps, n_features, self.d_model).transpose(1, 2)

        # Time attention block
        x_time = x.reshape(batch_size * n_features, time_steps, self.d_model)
        identity = x_time

        x_time, time_weights = self.time_attention(x_time, x_time, x_time)
        x_time = identity + self.dropout(x_time)
        x_time = self.time_norm(x_time)

        identity = x_time
        x_time = identity + self.dropout(self.time_ff(x_time))
        x_time = self.time_ff_norm(x_time)

        x = x_time.view(batch_size, n_features, time_steps, self.d_model)

        return x, feat_weights, time_weights

class DualAttentionModel(nn.Module):
    """
    A model that stacks multiple DualAttentionModules followed by a final projection.
    """
    def __init__(self, n_features, time_steps, d_model=128, n_heads=8, dropout=0.1, n_modules=1):
        super().__init__()
        
        # Stack of dual attention modules
        self.modules_list = nn.ModuleList([
            DualAttentionModule(n_features, time_steps, d_model, n_heads, dropout)
            for _ in range(n_modules)
        ])

        # Add learned pooling weights
        self.time_pool = nn.Parameter(torch.randn(d_model))
        self.pool_attention = nn.Linear(d_model, 1)
        self.feature_pool = nn.Linear(n_features, 1)  # Learned pooling weights
        
        # Output layers
        self.output_ff = FeedForward(d_model, dropout=dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.fcd = nn.Linear(d_model, d_model)
        self.final_fc = nn.Linear(d_model, 2)  # Final classification layer: number of classes
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, n_features, time_steps = x.shape
        x = x.unsqueeze(-1)  # [batch_size, n_features, time_steps, 1]
        
        # Store attention weights from all modules
        all_feat_weights = []
        all_time_weights = []

        # Pass through each dual attention module
        for i, module in enumerate(self.modules_list):
            x, feat_weights, time_weights = module(x, is_first_module=(i==0))
            all_feat_weights.append(feat_weights)
            all_time_weights.append(time_weights)

        # Pooling (over time)
        # Instead of: Global average pooling over time
        # x = x.mean(dim=2)  # [batch_size, n_features, d_model]
        # We do: learned pooling:
        # Calculate attention weights for time steps
        time_weights = torch.tanh(self.pool_attention(x))  # [batch_size, n_features, time_steps, 1]
        time_weights = F.softmax(time_weights, dim=2)  # [batch_size, n_features, time_steps, 1]
        # Apply weighted pooling
        x = (x * time_weights).sum(dim=2)  # [batch_size, n_features, d_model]

        # Final feed-forward with residual connection
        identity = x
        x = identity + self.dropout(self.output_ff(x))
        x = self.output_norm(x)

        # Final projection
        x = self.fcd(x)  # [batch_size, n_features, d_model]
        # Insted of mean pooling over features
        # x = x.mean(dim=1) # [batch_size, d_model] - average over features
        # Use learned pooling
        x = x.transpose(1, 2)  # [batch_size, d_model, n_features]
        x = self.feature_pool(x)  # [batch_size, d_model, 1]
        x = x.squeeze(-1)  # [batch_size, d_model]

        x = self.final_fc(x)  # [batch_size, 1] 
        
        # Return last module's attention weights (or could return all)
        return x, all_feat_weights[-1], all_time_weights[-1]

class DualAttentionModelOld(nn.Module):
    """
    A dual attention model that incorporates both feature and time attention mechanisms.
    
    Model architecture:
    - Input projection
    - Positional encoding   
    - Feature attention block
      - inlcudig feature feed-forward MLP
    - Time attention block
      - inlcudig time feed-forward MLP
    - Output layer
      - including final feed-forward MLP
    - Residual connections througout:
        - Feature attention block
        - Time attention block
        - Each feed-forward MLP block
    - To change: Global average pooling over time
    """
    def __init__(self, n_features, time_steps, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Feature attention block
        self.feature_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_ff = FeedForward(d_model, dropout=dropout)
        self.feature_ff_norm = nn.LayerNorm(d_model)

        # Time attention block
        self.time_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_ff = FeedForward(d_model, dropout=dropout)
        self.time_ff_norm = nn.LayerNorm(d_model)
        
        # Output layers with residual connection
        self.output_ff = FeedForward(d_model, dropout=dropout)
        self.output_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        # x shape: [batch_size, n_features, time_steps]
        batch_size, n_features, time_steps = x.shape

        # Add channel dimension and project
        x = x.unsqueeze(-1)  # [batch_size, n_features, time_steps, 1]
        x = self.input_projection(x)  # [batch_size, n_features, time_steps, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Feature attention block with residual connections
        # Reshape for feature attention
        x_feat = x.transpose(1, 2).reshape(batch_size * time_steps, n_features, self.d_model)
        identity = x_feat

        # Feature attention
        x_feat, feat_weights = self.feature_attention(x_feat, x_feat, x_feat)
        x_feat = identity + self.dropout(x_feat)  # First residual connection
        x_feat = self.feature_norm(x_feat)

        # Feature feed-forward
        identity = x_feat
        x_feat = identity + self.dropout(self.feature_ff(x_feat))  # Second residual connection
        x_feat = self.feature_ff_norm(x_feat)

        # Reshape back
        x = x_feat.view(batch_size, time_steps, n_features, self.d_model).transpose(1, 2)

        # Time attention block with residual connections
        # Reshape for time attention
        x_time = x.reshape(batch_size * n_features, time_steps, self.d_model)
        identity = x_time

        # Time attention
        x_time, time_weights = self.time_attention(x_time, x_time, x_time)
        x_time = identity + self.dropout(x_time)  # Third residual connection
        x_time = self.time_norm(x_time)

        # Time feed-forward
        identity = x_time
        x_time = identity + self.dropout(self.time_ff(x_time))  # Fourth residual connection
        x_time = self.time_ff_norm(x_time)

        # Reshape back
        x = x_time.view(batch_size, n_features, time_steps, self.d_model)

        # Global average pooling over time
        x = x.mean(dim=2)  # [batch_size, n_features, d_model]

        # Final feed-forward with residual connection
        identity = x
        x = identity + self.dropout(self.output_ff(x))  # Fifth residual connection
        x = self.output_norm(x)

        # Final projection
        x = self.fc(x).squeeze(-1)  # [batch_size, n_features]

        return x, feat_weights, time_weights


class DualAttentionClassifier1(nn.Module):
    """
    A dual attention model that incorporates both feature and time attention mechanisms.
    Similar to DualAttentionModel but outputs only 1 classification (for each sample).
    Uses DualAttentionModule for the main processing.
    """
    def __init__(self, n_features, time_steps, n_classes, d_model=128, n_heads=8, dropout=0.1, n_modules=1):
        super().__init__()
        
        # Stack of dual attention modules
        self.modules_list = nn.ModuleList([
            DualAttentionModule(n_features, time_steps, d_model, n_heads, dropout)
            for _ in range(n_modules)
        ])
        
        # Output layers with residual connection
        self.output_ff = FeedForward(d_model, dropout=dropout)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Learned pooling over features
        self.feature_pool = nn.Linear(n_features, 1)

        # Classification head (inlcuding feature pooling)
        self.classification_head = ClassificationHead(
            d_model=d_model,
            n_features=n_features,
            n_classes=n_classes,
            dropout=dropout,
            use_learned_pooling=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, n_features, time_steps]
        batch_size, n_features, time_steps = x.shape
        x = x.unsqueeze(-1)  # [batch_size, n_features, time_steps, 1]
        
        # Store attention weights from all modules
        all_feat_weights = []
        all_time_weights = []

        # Pass through each dual attention module
        for i, module in enumerate(self.modules_list):
            x, feat_weights, time_weights = module(x, is_first_module=(i==0))
            all_feat_weights.append(feat_weights)
            all_time_weights.append(time_weights)

        # Global average pooling over time
        x = x.mean(dim=2)  # [batch_size, n_features, d_model]

        # Final feed-forward with residual connection
        identity = x
        x = identity + self.dropout(self.output_ff(x))
        x = self.output_norm(x)

       # Classification head inlcuding feature pooling
        logits = self.classification_head(x)  # [batch_size, n_classes]
        
        return logits, all_feat_weights[-1], all_time_weights[-1]

class DualAttentionClassifier_old(nn.Module):
    """
    A dual attention model that incorporates both feature and time attention mechanisms.
    Similar to DualAttentionModelOld but outputs classifications for each feature (and for each sample ;-)).
    """
    def __init__(self, n_features, time_steps, n_classes, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features
        self.n_classes = n_classes

        # Input projection
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Feature attention block
        self.feature_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_ff = FeedForward(d_model, dropout=dropout)
        self.feature_ff_norm = nn.LayerNorm(d_model)

        # Time attention block
        self.time_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_ff = FeedForward(d_model, dropout=dropout)
        self.time_ff_norm = nn.LayerNorm(d_model)
        
        # Output layers with residual connection
        self.output_ff = FeedForward(d_model, dropout=dropout)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, n_features, time_steps]
        batch_size, n_features, time_steps = x.shape

        # Add channel dimension and project
        x = x.unsqueeze(-1)  # [batch_size, n_features, time_steps, 1]
        x = self.input_projection(x)  # [batch_size, n_features, time_steps, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Feature attention block
        x_feat = x.transpose(1, 2).reshape(batch_size * time_steps, n_features, self.d_model)
        identity = x_feat

        x_feat, feat_weights = self.feature_attention(x_feat, x_feat, x_feat)
        x_feat = identity + self.dropout(x_feat)
        x_feat = self.feature_norm(x_feat)

        identity = x_feat
        x_feat = identity + self.dropout(self.feature_ff(x_feat))
        x_feat = self.feature_ff_norm(x_feat)

        x = x_feat.view(batch_size, time_steps, n_features, self.d_model).transpose(1, 2)

        # Time attention block
        x_time = x.reshape(batch_size * n_features, time_steps, self.d_model)
        identity = x_time

        x_time, time_weights = self.time_attention(x_time, x_time, x_time)
        x_time = identity + self.dropout(x_time)
        x_time = self.time_norm(x_time)

        identity = x_time
        x_time = identity + self.dropout(self.time_ff(x_time))
        x_time = self.time_ff_norm(x_time)

        x = x_time.view(batch_size, n_features, time_steps, self.d_model)

        # Global average pooling over time
        x = x.mean(dim=2)  # [batch_size, n_features, d_model]

        # Final feed-forward with residual connection
        identity = x
        x = identity + self.dropout(self.output_ff(x))
        x = self.output_norm(x)

        # Classification head
        logits = self.classifier(x)  # [batch_size, n_features, n_classes]

        return logits, feat_weights, time_weights

class DualAttentionClassifier1(nn.Module):
    """
    A dual attention model that incorporates both feature and time attention mechanisms.
    Similar to DualAttentionModel but outputs only 1 classification (for each sample).
    """
    def __init__(self, n_features, time_steps, n_classes, d_model=128, n_heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.n_features = n_features
        self.n_classes = n_classes

        # Input projection
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Feature attention block
        self.feature_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_ff = FeedForward(d_model, dropout=dropout)
        self.feature_ff_norm = nn.LayerNorm(d_model)

        # Time attention block
        self.time_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_ff = FeedForward(d_model, dropout=dropout)
        self.time_ff_norm = nn.LayerNorm(d_model)
        
        # Output layers with residual connection
        self.output_ff = FeedForward(d_model, dropout=dropout)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Learned pooling over features
        self.feature_pool = nn.Linear(n_features, 1)

        # Classification head with feature pooling
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, n_features, time_steps]
        batch_size, n_features, time_steps = x.shape

        # Add channel dimension and project
        x = x.unsqueeze(-1)  # [batch_size, n_features, time_steps, 1]
        x = self.input_projection(x)  # [batch_size, n_features, time_steps, d_model]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Feature attention block
        x_feat = x.transpose(1, 2).reshape(batch_size * time_steps, n_features, self.d_model)
        identity = x_feat

        x_feat, feat_weights = self.feature_attention(x_feat, x_feat, x_feat)
        x_feat = identity + self.dropout(x_feat)
        x_feat = self.feature_norm(x_feat)

        identity = x_feat
        x_feat = identity + self.dropout(self.feature_ff(x_feat))
        x_feat = self.feature_ff_norm(x_feat)

        x = x_feat.view(batch_size, time_steps, n_features, self.d_model).transpose(1, 2)

        # Time attention block
        x_time = x.reshape(batch_size * n_features, time_steps, self.d_model)
        identity = x_time

        x_time, time_weights = self.time_attention(x_time, x_time, x_time)
        x_time = identity + self.dropout(x_time)
        x_time = self.time_norm(x_time)

        identity = x_time
        x_time = identity + self.dropout(self.time_ff(x_time))
        x_time = self.time_ff_norm(x_time)

        x = x_time.view(batch_size, n_features, time_steps, self.d_model)

        # Global average pooling over time
        x = x.mean(dim=2)  # [batch_size, n_features, d_model]

        # Final feed-forward with residual connection
        identity = x
        x = identity + self.dropout(self.output_ff(x))
        x = self.output_norm(x)

        # Pool across features using learned weights
        x = x.transpose(1, 2)  # [batch_size, d_model, n_features]
        x = self.feature_pool(x)  # [batch_size, d_model, 1]
        x = x.squeeze(-1)  # [batch_size, d_model]

        # Classification head
        logits = self.classifier(x)  # [batch_size, n_classes]

        return logits, feat_weights, time_weights

    
def visualize_predictions(model, test_loader, device='cuda', num_examples=3):
    
    def plot_predicted_vs_data(col):
        plt.scatter(predictions[:,col], y[:,col],
                    c=np.random.rand(3,), label=f'Feature {col}', alpha=0.5)
        # Add perfect line
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Feature {col}')
        plt.legend()
    
    model.eval()

    # Get some test examples
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)

    # Get predictions
    with torch.no_grad():
        predictions, feat_attn, time_attn = model(x)
        predictions = predictions.cpu().numpy()
        x = x.cpu().numpy()
        y = y.cpu().numpy()

    # Plot multiple features 
    for i, feature_idx in enumerate([0,1,2]):#([0, 7, 14]):  # Beginning, middle, and end features
        plt.figure(figsize=(15, 5))

        # Plot a few examples for each feature
        for example_idx in range(min(num_examples, x.shape[0])):
            plt.subplot(num_examples, 1, example_idx+1)

            # Plot input sequence
            time_input = np.arange(x.shape[2])
            plt.plot(time_input, x[example_idx, feature_idx],
                    label='Input Sequence', color='blue')

            # Plot true continuation
            #time_target = np.arange(x.shape[2], x.shape[2] + y.shape[2])
            time_target = [x.shape[2] + 5 - 1]
            plt.plot(time_target, y[example_idx, feature_idx], "x-",
                    label='True Continuation', color='green')

            # Plot prediction
            plt.scatter(time_target[0], predictions[example_idx, feature_idx],
                       color='red', label='Model Prediction', s=100)

            plt.title(f'Example {example_idx + 1}, Feature {feature_idx}')
            plt.legend(loc='upper left')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Plot data vs predicted & groundtruth vs predicted
    plt.figure(figsize=(15, 4))
    # data vs predicted
    plt.subplot(1, 3, 1)
    print(f"Data: {y.shape}")
    print(f"Predictions: {predictions.shape}")
    np.random.seed(1)
    for col in range(y.shape[1]):
        plot_predicted_vs_data(col)

    # Plot data vs predicted first features seperately
    plt.figure(figsize=(15,3))
    for f in range(4):
        plt.subplot(1, 4, f+1)
        plot_predicted_vs_data(f)

    # Visualize attention weights
    plt.figure(figsize=(15, 3))

    # Feature attention weights
    plt.subplot(1, 4, 1)
    feat_attn_avg = feat_attn.mean(dim=(0, 1)).cpu().numpy()
    plt.imshow(feat_attn_avg, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Feature Attention Weights')
    plt.xlabel('Target Feature')
    plt.ylabel('Source Feature')

    # Time attention weights
    plt.subplot(1, 4, 2)
    time_attn_avg = time_attn.mean(dim=(0, 1)).cpu().numpy()
    plt.imshow(time_attn_avg, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Time Attention Weights')
    plt.xlabel('Target Time Step')
    plt.ylabel('Source Time Step')

    plt.tight_layout()
    plt.show()

# Assuming you have lists/arrays of training metrics
def plot_training_curves(
    train_losses, 
    val_losses=None, 
    epochs=None,
    figsize=(10, 6)
):
    #plt.figure(figsize=figsize)
    epochs = epochs or range(1, len(train_losses) + 1)
    
    # Plot training loss
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    
    # Plot validation loss if available
    if val_losses:
        plt.plot(epochs, val_losses, 'r--', label='Validation Loss')
    
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()