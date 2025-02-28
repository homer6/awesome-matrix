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
    
    # Enable minor ticks and add grid lines
    plt.minorticks_on()
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    # Add row and column indices
    for i in range(matrix_np.shape[0]):
        for j in range(matrix_np.shape[1]):
            plt.text(j, i, f"{matrix_np[i, j]:.1f}", 
                     ha="center", va="center", 
                     color="black" if matrix_np[i, j] < 0.7 else "white")
    
    # Add dimension annotations with actual dimensions
    rows, cols = matrix_np.shape
    plt.title(f"{title}\nShape: {rows}×{cols}")
    plt.xlabel(f"Columns ({cols} columns)")
    plt.ylabel(f"Rows ({rows} rows)")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Let's visualize our matrices:

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
# Notice that the inner dimensions match: A has 3 columns and B has 3 rows.
# This means we can multiply these matrices together!

# %% [markdown]
# ## 1.4 Matrix Multiplication Preview
# 
# When multiplying matrices, we can only multiply if the inner dimensions match:
# - Matrix A with dimensions (m × n)
# - Matrix B with dimensions (n × p)
# 
# The result C = A @ B will have dimensions (m × p).
# 
# Let's visualize our matrices and their multiplication:

# %%
def visualize_matrix_multiplication(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Visualize matrix multiplication A @ B with dimensions.
    
    Args:
        A: First matrix (m × n)
        B: Second matrix (n × p)
    """
    # Check compatibility
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible dimensions: A is {A.shape}, B is {B.shape}")
    
    # Perform the multiplication
    C = A @ B
    
    # Create figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot matrices
    matrices = [A, B, C]
    titles = [
        f"Matrix A\n{A.shape[0]}×{A.shape[1]}", 
        f"Matrix B\n{B.shape[0]}×{B.shape[1]}",
        f"Result C = A @ B\n{C.shape[0]}×{C.shape[1]}"
    ]
    
    for i, (matrix, title) in enumerate(zip(matrices, titles)):
        matrix_np = matrix.detach().cpu().numpy()
        im = axs[i].imshow(matrix_np, cmap=cmap)
        axs[i].set_title(title)
        
        # Enable minor ticks
        axs[i].minorticks_on()
        axs[i].grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
        # Add text annotations
        for r in range(matrix_np.shape[0]):
            for c in range(matrix_np.shape[1]):
                axs[i].text(c, r, f"{matrix_np[r, c]:.1f}", 
                           ha="center", va="center", 
                           color="black" if matrix_np[r, c] < 0.7 else "white")
    
    # Add a shared colorbar
    fig.colorbar(im, ax=axs, shrink=0.6)
    
    # Add the operation text between plots
    plt.figtext(0.31, 0.5, "@", fontsize=24)
    plt.figtext(0.64, 0.5, "=", fontsize=24)
    
    # Add dimension explanation with actual dimensions
    m, n = A.shape
    n_check, p = B.shape
    plt.suptitle(f"Matrix Multiplication: ({m}×{n}) @ ({n_check}×{p}) → ({m}×{p})\n"
                f"The inner dimensions must match: {n} = {n_check}", fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return C

# %%
# Let's perform the matrix multiplication and visualize it
C = visualize_matrix_multiplication(A, B)

# %%
# Print the actual computation
print("Matrix multiplication result C = A @ B:")
print(C)

# Let's verify the dimensions
print(f"\nMatrix A shape: {A.shape}")
print(f"Matrix B shape: {B.shape}")
print(f"Result C shape: {C.shape}")
print("\nDimension rule: (m × n) @ (n × p) = (m × p)")
print(f"In our case: ({A.shape[0]} × {A.shape[1]}) @ ({B.shape[0]} × {B.shape[1]}) = ({C.shape[0]} × {C.shape[1]})")