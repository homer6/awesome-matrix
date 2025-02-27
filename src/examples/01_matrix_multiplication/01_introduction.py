# %% [markdown]
# # 1. Matrix Multiplication: Building Intuition
# 
# This example demonstrates matrix multiplication with PyTorch and 
# builds intuition for how dimensions work when multiplying matrices.
# 
# Matrix multiplication is a fundamental operation in linear algebra and forms the basis for:
# 
# - Neural network layers and transformations
# - 3D graphics transformations
# - Data transformations in machine learning
# - Solving systems of linear equations
# 
# In this tutorial, we'll explore matrix multiplication visually and build an intuitive understanding.

# %% [markdown]
# ## 1.1 Setup and Imports
# 
# First, we'll import the necessary libraries and set up our visualization tools.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Tuple, List

# Create a nicer colormap for visualizing matrices
colors = [(0.8, 0.8, 1), (0.1, 0.3, 0.8)]  # Light blue to darker blue
cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=100)

# %% [markdown]
# ## 1.2 Basic Matrix Creation
# 
# Let's create some simple matrices to work with:

# %%
# Create a 2×3 matrix (2 rows, 3 columns)
A = torch.tensor([[1., 2., 3.], 
                  [4., 5., 6.]])

# Create a 3×2 matrix (3 rows, 2 columns)
B = torch.tensor([[7., 8.], 
                  [9., 10.], 
                  [11., 12.]])

print(f"Matrix A shape: {A.shape} (2 rows × 3 columns)")
print(f"Matrix B shape: {B.shape} (3 rows × 2 columns)")

# Let's look at the actual contents
print("\nMatrix A:")
print(A)

print("\nMatrix B:")
print(B)

# %% [markdown]
# ## 1.3 Visualizing Matrices
# 
# To better understand matrices, we'll create a function to visualize them as heatmaps.

# %%
def visualize_matrix(matrix: torch.Tensor, title: str = "") -> None:
    """
    Visualize a matrix as a heatmap.
    
    Args:
        matrix: PyTorch tensor to visualize
        title: Optional title for the plot
    """
    # Convert to numpy for matplotlib
    matrix_np = matrix.detach().cpu().numpy()
    
    plt.figure(figsize=(7, 7))
    plt.imshow(matrix_np, cmap=cmap)
    plt.colorbar(shrink=0.8)
    
    # Add grid lines
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add row and column indices
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            plt.text(j, i, f"{matrix_np[i, j]:.1f}", 
                     ha="center", va="center", 
                     color="black" if matrix_np[i, j] < 0.7 else "white")
    
    # Add dimension annotations
    plt.title(f"{title}\nShape: {matrix_np.shape}")
    plt.xlabel(f"Columns (n={matrix_np.shape[1]})")
    plt.ylabel(f"Rows (m={matrix_np.shape[0]})")
    plt.tight_layout()
    plt.show()

# %%
# Visualize matrix A
visualize_matrix(A, "Matrix A")

# %%
# Visualize matrix B
visualize_matrix(B, "Matrix B")

# %% [markdown]
# The visualizations above help us see the structure of each matrix:
# 
# - Matrix A is 2×3 (2 rows, 3 columns)
# - Matrix B is 3×2 (3 rows, 2 columns)
# 
# In the next section, we'll explore how these matrices can be multiplied together.