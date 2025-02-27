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

### Option 1: Run Python Script

```bash
python src/examples/matrix_multiplication.py
```

### Option 2: Run Jupyter Notebook (Recommended)

For the best interactive experience with rich visualizations:

```bash
jupyter notebook src/examples/matrix_multiplication.ipynb
```

If the example runs without errors and displays visualizations, your setup is complete.

## Using Jupyter Notebooks (Recommended)

Examples are provided as both standalone Python scripts and interactive Jupyter notebooks. We highly recommend the notebook versions for better visualizations and interactivity.

To use Jupyter notebooks:

1. Start the Jupyter server:

   ```bash
   jupyter notebook
   ```

2. A browser window will open automatically. Navigate to the `src/examples` directory and click on any `.ipynb` file to open it

3. You can run each cell individually by selecting it and pressing `Shift+Enter`, or run the entire notebook via the "Run" menu

4. To make the most of the interactive visualizations, run each cell in sequence from top to bottom

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