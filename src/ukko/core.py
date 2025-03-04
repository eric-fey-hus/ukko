"""
Core functionality of the package.
Functions of the package are her. 
If you develop them in a notebook, mpve them here when ready.
"""

# attention model with residual connectoin
import torch
import torch.nn as nn
import math

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

class DualAttentionModel(nn.Module):
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

