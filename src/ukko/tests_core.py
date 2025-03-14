"""
Test suite for the core functionality.
"""

import pytest, torch
import ukko
from ukko.core import DualAttentionRegressor

def test_your_function():
    """Test your_function works as expected."""
    result = your_function(1, 2)
    assert result == 3

def test_model():
    # Example dimensions
    batch_size = 2
    n_features = 3
    time_steps = 5

    # Create random input data
    x = torch.randn(batch_size, n_features, time_steps)

    # Initialize model
    model = DualAttentionRegressor(
        n_features=n_features,
        time_steps=time_steps,
        d_model=10,
        n_heads=5,
        dropout=0.1
    )

    # Forward pass
    output, feat_attn, time_attn = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature attention weights shape: {feat_attn.shape}")
    print(f"Time attention weights shape: {time_attn.shape}")

    return model, output, feat_attn, time_attn

#if __name__ == "__main__":
#    model, output, feat_attn, time_attn = test_model()

def test_ClassificationHead_new():
    # For two binary labels
    head = ukko.core.ClassificationHead_new(
        d_model=128,
        n_features=10,
        n_labels=3,  # Three independent labels
        n_classes_per_label=2,  # Binary classification for each
        dropout=0.1,
        use_learned_pooling=True
    )

    # Example forward pass
    x = torch.randn(32, 10, 128)  # [batch_size, n_features, d_model]
    logits = head(x)  # Returns: [32, 3, 2] (batch_size, n_labels, n_classes_per_label)

    # 1. Check output shape
    assert logits.shape == (32, 3, 2), \
        f"Expected shape {(32, 3, 2)}, got {logits.shape}"

    # 2. Check output type
    assert isinstance(logits, torch.Tensor), "Output should be a torch.Tensor"

    # 3. Check value ranges
    assert torch.isfinite(logits).all(), "Output contains NaN or infinite values"

    # 4. Check specific tensor properties
    assert logits.dtype == torch.float32, f"Expected dtype float32, got {logits.dtype}"