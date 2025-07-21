# AI Coding Assistant Instructions for Ukko

## Project Overview
Ukko is a PyTorch-based library for analyzing tabular longitudinal data using transformer models. The core functionality includes:

- Dual attention mechanism for processing temporal and feature dimensions
- Support for both classification and regression tasks
- Integration with PyCoX for survival analysis
- Custom data handling for longitudinal data

## Key Components

### Data Processing (`src/ukko/data.py`)
- Input data should be formatted as 2D DataFrames with tuple column names like `"('feature1', timepoint1)"`
- Use `convert_to_3d_df()` to transform data into required 3D format (samples × features × timepoints)
- `SineWaveDataset` provides synthetic data generation for testing and examples

### Core Models (`src/ukko/core.py`)
- Base attention components: `PositionalEncoding`, `MultiHeadAttention`
- Main model architectures:
  - `DualAttentionClassifier`: Classification with dual attention
  - `DualAttentionRegressor`: Regression with dual attention
  - `DualAttentionRegressor1`: Single output regression

### Model Training Patterns
```python
# Standard training setup
model = DualAttentionRegressor(
    n_features=10,
    time_steps=50,
    d_model=128,
    n_heads=8
)

# Convert numpy arrays to tensors
x_train = torch.tensor(input_data, dtype=torch.float32)
y_train = torch.tensor(target_data, dtype=torch.float32)

# Create dataset and dataloader
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### Survival Analysis Integration
- Extend `pycox.models.LogisticHazard` for custom survival analysis
- Use `CustomLogisticHazard` for direct numpy array handling

## Development Workflows

### Testing
- Run tests from project root: `python -m pytest tests/`
- Use `SineWaveDataset` for integration testing
- Test files follow pattern: `test_*.py` in `tests/` directory

### Model Development
1. Prototype in notebooks (e.g., `Survival_model_dev.ipynb`)
2. Move stable code to `src/ukko/core.py`
3. Add tests in `tests/` directory

## Project Conventions
- Model outputs include attention weights for interpretability
- Use PyTorch's nn.Module as base class for all models
- Maintain shape documentation in docstrings: `[batch_size, n_features, time_steps]`
- Save best models with timestamp: `weight_checkpoint_{date}_{id}.pt`
