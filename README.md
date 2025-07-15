# ukko
Transformer model for tabular longitudinal data.

The ukko codebase is a Python package for deep learning on tabular longitudinal (time-series) and survival data, with a focus on transformer-based and Cox proportional hazards models. 

- Several survival heads (survival model output layers) are beeing explored atm
- Key feature of teh core models is a dual attention mechanism 

Running notes for development (try to keep this up to date!):
- Survival heads:
  - Own survival head implementation:  
    `Survival_model_dev`
  - torchsurv heads:  
    `torchsurv_AML_model`

## Structure

```sh
ukko/
├── pyproject.toml
├── src/
│   └── ukko/
│       ├── __init__.py
│       └── core.py
└── ukko_get_started.ipynb
```

- pyproject.toml:
  - Configures the project metadata (name, version, author, description, etc.).
  - Uses hatchling as the build backend.
  - Specifies the source code is under src/ukko.
  - Includes build instructions for both wheel and sdist distributions.

- src/ukko/:
  - This is the main package directory.
  - `__init__.py`: Initializes the package and imports everything from core.py.
  - `core.py`: Function definitions of the package. 
  - `ukko_get_started.ipynb`:
     A Jupyter notebook containing usage examplesfor the ukko package.

## Key Features

- Editable Install:  
  The project is set up for development with editable installs (pip install -e .), making it easy to test changes.

- Modern Packaging:  
  Uses pyproject.toml and the src layout, which is recommended for modern Python projects.

- Intended Usage:  
  The package is designed for transformer models applied to tabular longitudinal data, as described in the project metadata.