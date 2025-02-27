# Installation Guide

This guide will help you set up the environment to run and explore the matrix operation examples in this repository.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (to clone the repository)

## Installation Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/homer6/awesome-matrix.git
   cd awesome-matrix
   ```

2. **Set up a virtual environment (recommended)**

   ```bash
   # Create a virtual environment
   python -m venv venv

   # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This will install all required packages including:
   - PyTorch (for tensor operations)
   - NumPy (for numerical computing)
   - Matplotlib (for visualizations)
   - Pandas (for data manipulation)
   - Jupyter (for interactive notebooks)
   - scikit-learn (for machine learning examples)
   - ipywidgets (for interactive visualizations)

## Verifying Installation

To verify that everything is installed correctly, you can run a simple example:

```bash
python src/examples/matrix_multiplication.py
```

If the example runs without errors and displays visualizations, your setup is complete.

## Running Jupyter Notebooks

Some examples may be provided as Jupyter notebooks. To run these:

1. Start the Jupyter server:

   ```bash
   jupyter notebook
   ```

2. Navigate to the notebook file (.ipynb) in the browser window that opens

## Troubleshooting

- **PyTorch Installation Issues**: If you encounter problems with PyTorch, visit the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for platform-specific instructions.

- **GPU Support**: To utilize GPU acceleration (if available):
  ```bash
  # Install PyTorch with CUDA support (example for CUDA 11.8)
  pip install torch --index-url https://download.pytorch.org/whl/cu118
  ```

- **Matplotlib Display Issues**: If visualizations don't display properly, ensure you have a backend configured for your environment. In headless environments, you may need to set:
  ```python
  import matplotlib
  matplotlib.use('Agg')  # For non-interactive environments
  ```

## Next Steps

Once installation is complete, check out the [examples directory](src/examples/README.md) for available demonstrations and tutorials.