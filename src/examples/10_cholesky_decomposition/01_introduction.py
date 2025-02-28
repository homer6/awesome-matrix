# %% [markdown]
# # Cholesky Decomposition: Introduction
# 
# The Cholesky decomposition is a method for decomposing a Hermitian, positive-definite matrix into 
# the product of a lower triangular matrix and its conjugate transpose:
# 
# $$A = LL^*$$
# 
# where $L$ is a lower triangular matrix with real and positive diagonal entries, and $L^*$ is the 
# conjugate transpose of $L$. For real matrices, this simplifies to:
# 
# $$A = LL^T$$
# 
# This decomposition is particularly useful because:
# - It's computationally efficient (roughly half the cost of LU decomposition)
# - It's numerically stable when A is positive definite
# - It has applications in linear systems, Monte Carlo simulations, and optimization

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the default style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# %% [markdown]
# ## Creating a Positive Definite Matrix
# 
# For a Cholesky decomposition to exist, the input matrix must be positive definite.
# A symmetric matrix is positive definite if all its eigenvalues are positive.
# 
# One way to ensure a matrix is positive definite is to create a matrix $A = BB^T$ 
# where $B$ is any non-singular matrix.

# %%
def create_positive_definite_matrix(n, method="random"):
    """Create a positive definite matrix of size n×n."""
    if method == "random":
        # Create a random matrix
        B = torch.randn(n, n)
        # A = B*B^T is guaranteed to be positive definite (if B is full rank)
        A = B @ B.T
        # Add a small value to the diagonal to ensure positive definiteness
        A = A + torch.eye(n) * 1e-5
        return A
    elif method == "predetermined":
        # A predefined example
        A = torch.tensor([[4.0, 1.0, 1.0], 
                           [1.0, 3.0, 2.0], 
                           [1.0, 2.0, 6.0]])
        return A
    else:
        raise ValueError("Unknown method")

# Create a 3×3 positive definite matrix
A = create_positive_definite_matrix(3, method="predetermined")
print("Matrix A:")
print(A)

# %% [markdown]
# Let's confirm that matrix A is positive definite by checking its eigenvalues:

# %%
eigenvalues = torch.linalg.eigvalsh(A)
print("Eigenvalues of A:", eigenvalues)
print("Is A positive definite?", torch.all(eigenvalues > 0).item())

# %% [markdown]
# ## Computing the Cholesky Decomposition
# 
# PyTorch provides the `torch.linalg.cholesky` function to compute the Cholesky decomposition.
# It returns the lower triangular matrix $L$ such that $A = LL^T$.

# %%
L = torch.linalg.cholesky(A)
print("Cholesky factor L:")
print(L)

# Verify that A = L*L^T
reconstructed_A = L @ L.T
print("\nReconstructed A = L*L^T:")
print(reconstructed_A)

print("\nError in reconstruction:", torch.norm(A - reconstructed_A).item())

# %% [markdown]
# ## Visualizing the Decomposition
# 
# Let's visualize the original matrix and its Cholesky decomposition to build intuition.

# %%
def plot_matrix(matrix, title):
    """Plot a matrix as a heatmap with annotations."""
    plt.figure(figsize=(7, 6))
    
    # Create heatmap
    ax = sns.heatmap(matrix.numpy(), annot=True, fmt=".2f", cmap="Blues",
                    linewidths=.5, cbar=True)
    
    # Add column and row indices
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(range(matrix.shape[1]))
    ax.set_yticklabels(range(matrix.shape[0]))
    
    # Add labels and title
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.title(title)
    
    plt.tight_layout()
    return ax

# Visualize the original matrix and its Cholesky decomposition
plot_matrix(A, f"Original Matrix A ({A.shape[0]}×{A.shape[1]})")
plt.show()

plot_matrix(L, f"Cholesky Factor L ({L.shape[0]}×{L.shape[1]})")
plt.show()

# %% [markdown]
# ## Understanding the Algorithm
# 
# The Cholesky decomposition can be computed using a step-by-step algorithm. 
# For an $n \times n$ matrix, the elements of $L$ can be found as:
# 
# $$L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}$$
# 
# $$L_{ji} = \frac{1}{L_{ii}} \left( A_{ji} - \sum_{k=1}^{i-1} L_{jk} L_{ik} \right)$$
# 
# where $i < j$ and $1 \leq i \leq n$.
# 
# Let's implement this algorithm to understand it better:

# %%
def cholesky_decomposition(A):
    """
    Compute the Cholesky decomposition of a positive definite matrix A.
    
    Parameters:
        A (torch.Tensor): A positive definite matrix
        
    Returns:
        L (torch.Tensor): Lower triangular matrix such that A = L*L^T
    """
    n = A.shape[0]
    # Initialize L as zeros
    L = torch.zeros_like(A)
    
    # Perform the Cholesky decomposition
    for i in range(n):
        # Compute diagonal elements
        sum_sq = torch.sum(L[i, :i]**2)
        L[i, i] = torch.sqrt(A[i, i] - sum_sq)
        
        # Compute non-diagonal elements in the current column
        for j in range(i+1, n):
            sum_prod = torch.sum(L[j, :i] * L[i, :i])
            L[j, i] = (A[j, i] - sum_prod) / L[i, i]
    
    return L

# Compute the Cholesky decomposition using our implementation
L_manual = cholesky_decomposition(A)
print("Cholesky factor L (manual implementation):")
print(L_manual)

# Compare with PyTorch's implementation
print("\nDifference between manual and PyTorch implementation:")
print(torch.norm(L - L_manual).item())

# %% [markdown]
# ## Step-by-Step Visualization
# 
# Let's visualize how the Cholesky decomposition builds the L matrix step by step:

# %%
def visualize_cholesky_steps(A):
    """Visualize the Cholesky decomposition step by step."""
    n = A.shape[0]
    # Initialize L as zeros
    L = torch.zeros_like(A)
    
    plt.figure(figsize=(15, 5 * n))
    
    # Plot original matrix
    plt.subplot(n+1, 3, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title("Original Matrix A")
    
    step = 2
    # Perform the Cholesky decomposition step by step
    for i in range(n):
        # Compute diagonal elements
        sum_sq = torch.sum(L[i, :i]**2)
        L[i, i] = torch.sqrt(A[i, i] - sum_sq)
        
        # Plot after computing diagonal
        plt.subplot(n+1, 3, step)
        sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5, 
                    mask=(L == 0).numpy() & ~torch.eye(n, dtype=bool).numpy())
        plt.title(f"Step {i+1}.1: Compute L[{i},{i}]")
        step += 1
        
        # Compute non-diagonal elements in the current column
        for j in range(i+1, n):
            sum_prod = torch.sum(L[j, :i] * L[i, :i])
            L[j, i] = (A[j, i] - sum_prod) / L[i, i]
            
            # Plot after each non-diagonal element
            plt.subplot(n+1, 3, step)
            sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5,
                        mask=(L == 0).numpy() & ~torch.eye(n, dtype=bool).numpy())
            plt.title(f"Step {i+1}.{j-i+1}: Compute L[{j},{i}]")
            step += 1
    
    # Final result
    plt.subplot(n+1, 3, step)
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title("Final Cholesky Factor L")
    
    plt.tight_layout()
    plt.show()
    
    return L

# Visualize the step-by-step Cholesky decomposition
visualize_cholesky_steps(A)

# %% [markdown]
# ## Computational Complexity
# 
# The Cholesky decomposition requires approximately $\frac{1}{3}n^3$ floating-point operations for an $n \times n$ matrix, which is roughly half the cost of LU decomposition. This efficiency makes it preferable when applicable.
# 
# Let's measure the time it takes to compute Cholesky decomposition for matrices of different sizes:

# %%
import time

sizes = [10, 50, 100, 500, 1000]
cholesky_times = []

for n in sizes:
    # Create a positive definite matrix
    A = create_positive_definite_matrix(n)
    
    # Measure time for Cholesky decomposition
    start_time = time.time()
    L = torch.linalg.cholesky(A)
    end_time = time.time()
    
    cholesky_times.append(end_time - start_time)
    
    print(f"Size {n}×{n}: {cholesky_times[-1]:.6f} seconds")

# Plot the time complexity
plt.figure(figsize=(10, 6))
plt.plot(sizes, cholesky_times, 'o-', linewidth=2)
plt.xlabel('Matrix Size (n)')
plt.ylabel('Computation Time (seconds)')
plt.title('Cholesky Decomposition Time Complexity')
plt.grid(True)
plt.show()

# %% [markdown]
# ## Properties of Cholesky Decomposition
# 
# The Cholesky decomposition has several important properties:
# 
# 1. **Uniqueness**: For a given positive definite matrix A, the Cholesky decomposition L is unique.
# 2. **Numerical Stability**: The decomposition is numerically stable for positive definite matrices.
# 3. **Efficiency**: It requires approximately half the operations of LU decomposition.
# 4. **Connection to LU**: If A = LU is the LU decomposition of A, and A is symmetric positive definite, then L = U^T.
# 5. **Determinant**: The determinant of A can be efficiently computed as the square of the product of the diagonal elements of L.

# %%
# Demonstrating determinant calculation using Cholesky
det_A = torch.det(A)
det_using_cholesky = torch.prod(torch.diag(L))**2

print(f"Determinant of A using torch.det: {det_A.item():.6f}")
print(f"Determinant of A using Cholesky: {det_using_cholesky.item():.6f}")

# %% [markdown]
# ## Summary
# 
# In this introduction to Cholesky decomposition, we have:
# 
# 1. Learned the mathematical definition of Cholesky decomposition
# 2. Implemented and visualized the decomposition algorithm
# 3. Explored the computational complexity of the algorithm
# 4. Discussed key properties of the decomposition
# 
# In the next notebook, we'll explore more algorithms and methods related to Cholesky decomposition, including modified Cholesky decomposition and how to handle positive semi-definite matrices.