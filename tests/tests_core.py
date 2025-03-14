"""
Test suite for the core functionality.
"""

import pytest, torch
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