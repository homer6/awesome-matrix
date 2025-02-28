# %% [markdown]
# # LU Decomposition: Introduction
# 
# LU decomposition (or LU factorization) is a fundamental matrix decomposition technique that expresses a matrix as the product of a lower triangular matrix (L) and an upper triangular matrix (U). It serves as the computational foundation for various numerical algorithms, particularly for solving systems of linear equations.
# 
# In this notebook, we will:
# 
# 1. Understand the concept of LU decomposition
# 2. Implement LU decomposition from scratch
# 3. Visualize the decomposition process
# 4. Compare with built-in functions
# 
# LU decomposition is closely related to Gaussian elimination, which is the process of transforming a matrix into row echelon form (upper triangular). The L matrix captures the multipliers used during this process.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import time
import scipy.linalg

# For better looking plots
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a custom colormap (light blue to dark blue)
colors = [(0.95, 0.95, 1), (0.0, 0.2, 0.6)]
blue_cmap = LinearSegmentedColormap.from_list('CustomBlue', colors, N=100)

# %% [markdown]
# ## Basic Concept of LU Decomposition
# 
# For a square matrix $A$, the LU decomposition finds matrices $L$ and $U$ such that:
# 
# $A = LU$
# 
# where:
# - $L$ is a lower triangular matrix (all elements above the main diagonal are zero)
# - $U$ is an upper triangular matrix (all elements below the main diagonal are zero)
# 
# In many cases, we also set the diagonal elements of $L$ to 1, which makes the decomposition unique.
# 
# Let's first create a simple example to demonstrate the concept:

# %%
def create_example_matrix(n=3, method="random"):
    """Create a matrix for LU decomposition demonstration."""
    if method == "random":
        # Create a random matrix
        A = torch.rand(n, n) * 10
        # Make it diagonally dominant for numerical stability
        for i in range(n):
            A[i, i] = torch.sum(torch.abs(A[i, :])) + 1.0
    elif method == "simple":
        # Create a simple matrix with known decomposition
        A = torch.tensor([
            [2.0, 1.0, 1.0],
            [4.0, 3.0, 3.0],
            [8.0, 7.0, 9.0]
        ])
    else:
        raise ValueError("Unknown method")
    
    return A

# Create a simple example matrix
A_simple = create_example_matrix(method="simple")

# Display the matrix
def plot_matrix(matrix, title="Matrix", annotate=True, cmap=blue_cmap):
    """Plot a matrix as a heatmap with annotations."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
        
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix_np, annot=annotate, fmt=".2f", cmap=cmap, 
                    linewidths=1, cbar=True)
    plt.title(title)
    
    # Add row and column indices
    ax.set_xticks(np.arange(matrix_np.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix_np.shape[0]) + 0.5)
    ax.set_xticklabels([f"Col {i+1}" for i in range(matrix_np.shape[1])])
    ax.set_yticklabels([f"Row {i+1}" for i in range(matrix_np.shape[0])])
    
    plt.tight_layout()
    plt.show()

plot_matrix(A_simple, "Example Matrix A")

# %% [markdown]
# ## LU Decomposition Algorithm
# 
# The LU decomposition can be computed using the Gaussian elimination process. The key idea is to:
# 
# 1. Iteratively transform the matrix $A$ into an upper triangular matrix $U$ using elementary row operations
# 2. Keep track of the multipliers used in each step to form the lower triangular matrix $L$
# 
# Let's implement the algorithm from scratch:

# %%
def lu_decomposition(A, pivot=False):
    """
    Perform LU decomposition on matrix A.
    
    Args:
        A: Input matrix as a PyTorch tensor
        pivot: Whether to use partial pivoting (PA = LU)
    
    Returns:
        L: Lower triangular matrix
        U: Upper triangular matrix
        P: Permutation matrix (if pivot=True)
    """
    n = A.shape[0]
    
    # Make a copy to avoid modifying the original matrix
    A_work = A.clone()
    
    # Initialize L as identity matrix
    L = torch.eye(n, dtype=A.dtype)
    
    # Initialize permutation matrix (for pivoting)
    P = torch.eye(n, dtype=A.dtype)
    
    # Perform Gaussian elimination
    for k in range(n-1):  # Loop through each column
        # Partial pivoting (optional)
        if pivot:
            # Find the index of the maximum absolute value in the current column (from k to n)
            max_idx = torch.argmax(torch.abs(A_work[k:, k])) + k
            
            # If the max is not at the current row, swap rows
            if max_idx != k:
                # Swap rows in A_work
                A_work[[k, max_idx], :] = A_work[[max_idx, k], :]
                
                # Swap rows in P (up to the current column)
                P[[k, max_idx], :] = P[[max_idx, k], :]
                
                # Swap rows in L (up to the current column)
                if k > 0:
                    L[[k, max_idx], :k] = L[[max_idx, k], :k]
        
        # Skip if the current pivot is zero (singular matrix)
        if torch.abs(A_work[k, k]) < 1e-10:
            continue
        
        # For each row below the current row
        for i in range(k+1, n):
            # Compute the multiplier
            factor = A_work[i, k] / A_work[k, k]
            L[i, k] = factor
            
            # Update the current row by subtracting the scaled pivot row
            A_work[i, k:] -= factor * A_work[k, k:]
    
    # The resulting A_work is now the upper triangular matrix U
    U = A_work
    
    if pivot:
        return P, L, U
    else:
        return L, U

# Test the LU decomposition on our example matrix
L_simple, U_simple = lu_decomposition(A_simple)

# Display the results
plot_matrix(L_simple, "Lower Triangular Matrix (L)")
plot_matrix(U_simple, "Upper Triangular Matrix (U)")

# Check that A = LU
A_reconstructed = L_simple @ U_simple
plot_matrix(A_reconstructed, "Reconstructed Matrix (L×U)")

# Calculate reconstruction error
reconstruction_error = torch.norm(A_simple - A_reconstructed).item()
print(f"Reconstruction error: {reconstruction_error:.2e}")

# %% [markdown]
# ### Visualizing the Decomposition
# 
# Let's visualize the LU decomposition process more intuitively by showing the step-by-step Gaussian elimination:

# %%
def visualize_lu_decomposition(A):
    """Visualize the step-by-step process of LU decomposition."""
    n = A.shape[0]
    
    # Make a copy to avoid modifying the original matrix
    A_work = A.clone()
    
    # Initialize L as identity matrix
    L = torch.eye(n, dtype=A.dtype)
    
    plt.figure(figsize=(15, 5 * n))
    
    # Plot the original matrix
    plt.subplot(n + 1, 3, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Original Matrix A")
    
    # Plot initial L and U
    plt.subplot(n + 1, 3, 2)
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Initial L (Identity)")
    
    plt.subplot(n + 1, 3, 3)
    sns.heatmap(A_work.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Initial U (Copy of A)")
    
    # Perform Gaussian elimination step by step
    for k in range(n-1):  # Loop through each column
        # For each row below the current row
        for i in range(k+1, n):
            # Compute the multiplier
            factor = A_work[i, k] / A_work[k, k]
            L[i, k] = factor
            
            # Update the current row by subtracting the scaled pivot row
            A_work[i, k:] -= factor * A_work[k, k:]
        
        # Plot the current state
        plt.subplot(n + 1, 3, (k + 2) * 3 - 2)
        sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title(f"Original Matrix A (Step {k+1})")
        
        plt.subplot(n + 1, 3, (k + 2) * 3 - 1)
        sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title(f"L Matrix (Step {k+1})")
        
        plt.subplot(n + 1, 3, (k + 2) * 3)
        sns.heatmap(A_work.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title(f"U Matrix (Step {k+1})")
    
    # The final result
    plt.subplot(n + 1, 3, n * 3 + 1)
    reconstructed = L @ A_work
    sns.heatmap(reconstructed.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Reconstructed A = LU")
    
    plt.subplot(n + 1, 3, n * 3 + 2)
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Final L")
    
    plt.subplot(n + 1, 3, n * 3 + 3)
    sns.heatmap(A_work.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Final U")
    
    plt.tight_layout()
    plt.show()
    
    return L, A_work

# Visualize the LU decomposition process on our example matrix
L_visual, U_visual = visualize_lu_decomposition(A_simple)

# %% [markdown]
# ## Using Built-in LU Decomposition Functions
# 
# Both NumPy and PyTorch provide built-in functions for LU decomposition. Let's use them and compare the results with our implementation:

# %%
def compare_lu_implementations(A):
    """Compare different LU decomposition implementations."""
    # Our implementation
    start_time = time.time()
    L_ours, U_ours = lu_decomposition(A)
    our_time = time.time() - start_time
    
    # NumPy implementation (using SciPy)
    A_np = A.numpy()
    start_time = time.time()
    P_np, L_np, U_np = scipy.linalg.lu(A_np)
    np_time = time.time() - start_time
    
    # PyTorch implementation
    start_time = time.time()
    LU, pivots = torch.linalg.lu_factor(A)
    # PyTorch's output is a bit different; we need to extract L and U
    n = A.shape[0]
    U_torch = torch.triu(LU)
    L_torch = torch.tril(LU, -1) + torch.eye(n)
    torch_time = time.time() - start_time
    
    # Convert to PyTorch tensors for consistency
    L_np_tensor = torch.from_numpy(L_np)
    U_np_tensor = torch.from_numpy(U_np)
    
    # Calculate reconstruction errors
    our_error = torch.norm(A - L_ours @ U_ours).item()
    np_error = torch.norm(A - torch.from_numpy(P_np.T) @ L_np_tensor @ U_np_tensor).item()
    torch_error = torch.norm(A - L_torch @ U_torch).item()
    
    # Plot the results
    plt.figure(figsize=(15, 12))
    
    # Our implementation
    plt.subplot(3, 3, 1)
    sns.heatmap(L_ours.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Our L")
    
    plt.subplot(3, 3, 2)
    sns.heatmap(U_ours.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Our U")
    
    plt.subplot(3, 3, 3)
    reconstructed = L_ours @ U_ours
    sns.heatmap(reconstructed.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title(f"Our LU\nError: {our_error:.2e}, Time: {our_time:.4f}s")
    
    # NumPy implementation
    plt.subplot(3, 3, 4)
    sns.heatmap(L_np, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("NumPy L")
    
    plt.subplot(3, 3, 5)
    sns.heatmap(U_np, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("NumPy U")
    
    plt.subplot(3, 3, 6)
    reconstructed_np = P_np.T @ L_np @ U_np
    sns.heatmap(reconstructed_np, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title(f"NumPy PLU\nError: {np_error:.2e}, Time: {np_time:.4f}s")
    
    # PyTorch implementation
    plt.subplot(3, 3, 7)
    sns.heatmap(L_torch.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("PyTorch L")
    
    plt.subplot(3, 3, 8)
    sns.heatmap(U_torch.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("PyTorch U")
    
    plt.subplot(3, 3, 9)
    reconstructed_torch = L_torch @ U_torch
    sns.heatmap(reconstructed_torch.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title(f"PyTorch LU\nError: {torch_error:.2e}, Time: {torch_time:.4f}s")
    
    plt.tight_layout()
    plt.show()
    
    # Print the performance comparison
    print("Performance Comparison:")
    print(f"Our Implementation:  Error = {our_error:.2e}, Time = {our_time:.4f}s")
    print(f"NumPy Implementation: Error = {np_error:.2e}, Time = {np_time:.4f}s")
    print(f"PyTorch Implementation: Error = {torch_error:.2e}, Time = {torch_time:.4f}s")

# Compare different LU implementations on our example matrix
compare_lu_implementations(A_simple)

# Create and test a larger random matrix
A_random = create_example_matrix(n=5, method="random")
compare_lu_implementations(A_random)

# %% [markdown]
# ## The Need for Pivoting
# 
# In the basic LU decomposition, we assume that we can use the diagonal element as a pivot for each step of Gaussian elimination. However, if the pivot element is zero or very small, this can lead to numerical instability.
# 
# **Partial pivoting** addresses this issue by selecting the largest absolute value in the current column as the pivot. This results in a permutation of the rows, so we get $PA = LU$ where $P$ is a permutation matrix.
# 
# Let's create a matrix that requires pivoting and demonstrate the difference:

# %%
def create_pivoting_example():
    """Create an example matrix that requires pivoting."""
    # Create a matrix where the first pivot would be small or zero
    A = torch.tensor([
        [0.001, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0]
    ])
    
    return A

A_pivot = create_pivoting_example()
plot_matrix(A_pivot, "Matrix Requiring Pivoting")

# %% [markdown]
# ### LU Decomposition Without Pivoting

# %%
def visualize_pivoting_comparison(A):
    """Visualize LU decomposition with and without pivoting."""
    n = A.shape[0]
    
    # LU without pivoting
    try:
        L_no_pivot, U_no_pivot = lu_decomposition(A, pivot=False)
        no_pivot_error = torch.norm(A - L_no_pivot @ U_no_pivot).item()
        no_pivot_failed = False
    except Exception as e:
        print(f"LU without pivoting failed: {e}")
        no_pivot_failed = True
    
    # LU with pivoting
    P_pivot, L_pivot, U_pivot = lu_decomposition(A, pivot=True)
    pivot_error = torch.norm(P_pivot @ A - L_pivot @ U_pivot).item()
    
    # Plot the results
    if no_pivot_failed:
        plt.figure(figsize=(12, 4))
        cols = 3
    else:
        plt.figure(figsize=(12, 8))
        cols = 3
        rows = 2
    
    # Original matrix
    plt.subplot(rows if not no_pivot_failed else 1, cols, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Original Matrix A")
    
    if not no_pivot_failed:
        # Without pivoting
        plt.subplot(rows, cols, 2)
        sns.heatmap(L_no_pivot.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title("L (No Pivoting)")
        
        plt.subplot(rows, cols, 3)
        sns.heatmap(U_no_pivot.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title("U (No Pivoting)")
        
        plt.subplot(rows, cols, 4)
        reconstructed_no_pivot = L_no_pivot @ U_no_pivot
        sns.heatmap(reconstructed_no_pivot.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title(f"LU (No Pivoting)\nError: {no_pivot_error:.2e}")
    
    # With pivoting
    base_idx = 4 if not no_pivot_failed else 2
    plt.subplot(rows if not no_pivot_failed else 1, cols, base_idx)
    sns.heatmap(P_pivot.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("P (With Pivoting)")
    
    plt.subplot(rows if not no_pivot_failed else 1, cols, base_idx + 1)
    sns.heatmap(L_pivot.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("L (With Pivoting)")
    
    plt.subplot(rows if not no_pivot_failed else 1, cols, base_idx + 2)
    sns.heatmap(U_pivot.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("U (With Pivoting)")
    
    if not no_pivot_failed:
        plt.subplot(rows, cols, base_idx + 3)
        reconstructed_pivot = L_pivot @ U_pivot
        sns.heatmap((P_pivot @ A).numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
        plt.title(f"PA (With Pivoting)")
    
    plt.tight_layout()
    plt.show()
    
    # Print the performance comparison
    print("Performance Comparison:")
    if not no_pivot_failed:
        print(f"Without Pivoting: Error = {no_pivot_error:.2e}")
    print(f"With Pivoting: Error = {pivot_error:.2e}")

# Visualize the effect of pivoting
visualize_pivoting_comparison(A_pivot)

# %% [markdown]
# ## Real-World Example: 3D Grid Problem
# 
# Let's consider a practical application: solving a 3D Poisson equation on a grid. This kind of problem appears in physics simulations, fluid dynamics, and many other fields.
# 
# The system of equations that arises from discretizing a 3D Poisson equation often has a specific structure, and LU decomposition can be an efficient way to solve it.

# %%
def create_3d_grid_matrix(nx=4, ny=4, nz=4):
    """Create a matrix representing a 3D grid problem."""
    n = nx * ny * nz  # Total number of grid points
    A = torch.zeros((n, n))
    
    # Fill the matrix based on a 7-point stencil (central difference)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = i + j * nx + k * nx * ny  # Flattened 3D index
                
                # Diagonal element (center point)
                A[idx, idx] = 6.0
                
                # Neighbors in x-direction
                if i > 0:
                    A[idx, idx-1] = -1.0
                if i < nx-1:
                    A[idx, idx+1] = -1.0
                
                # Neighbors in y-direction
                if j > 0:
                    A[idx, idx-nx] = -1.0
                if j < ny-1:
                    A[idx, idx+nx] = -1.0
                
                # Neighbors in z-direction
                if k > 0:
                    A[idx, idx-nx*ny] = -1.0
                if k < nz-1:
                    A[idx, idx+nx*ny] = -1.0
    
    return A

# Create a small 3D grid matrix for visualization
A_3d_small = create_3d_grid_matrix(nx=2, ny=2, nz=2)
plot_matrix(A_3d_small, "3D Grid Matrix (2×2×2)")

# Create a larger matrix for performance testing
A_3d = create_3d_grid_matrix(nx=4, ny=4, nz=4)

# %% [markdown]
# ### Visualizing the Sparsity Pattern
# 
# The matrix from a 3D grid problem has a specific sparsity pattern. Let's visualize it:

# %%
def plot_sparsity(A, title="Sparsity Pattern"):
    """Plot the sparsity pattern of a matrix."""
    if isinstance(A, torch.Tensor):
        A_np = A.numpy()
    else:
        A_np = A
    
    plt.figure(figsize=(10, 8))
    plt.spy(A_np, marker='.', markersize=2)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    # Calculate sparsity
    n_elements = A_np.size
    n_nonzero = np.count_nonzero(A_np)
    sparsity = 1.0 - (n_nonzero / n_elements)
    
    print(f"Matrix size: {A_np.shape}")
    print(f"Number of non-zero elements: {n_nonzero}")
    print(f"Sparsity: {sparsity:.2%}")

# Visualize the sparsity pattern
plot_sparsity(A_3d, "3D Grid Matrix Sparsity Pattern")

# %% [markdown]
# ### Solving a Linear System with LU Decomposition
# 
# Now let's solve a linear system $Ax = b$ using LU decomposition:
# 
# 1. Decompose $A = LU$
# 2. Solve $Ly = b$ for $y$ (forward substitution)
# 3. Solve $Ux = y$ for $x$ (backward substitution)
# 
# This is more efficient than directly inverting the matrix, especially for large matrices.

# %%
def solve_with_lu(A, b, use_pivoting=True):
    """Solve a linear system Ax = b using LU decomposition."""
    n = A.shape[0]
    
    # Perform LU decomposition
    if use_pivoting:
        P, L, U = lu_decomposition(A, pivot=True)
        b_permuted = P @ b
    else:
        L, U = lu_decomposition(A, pivot=False)
        b_permuted = b
    
    # Forward substitution to solve Ly = b
    y = torch.zeros_like(b)
    for i in range(n):
        y[i] = b_permuted[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        # No need to divide by L[i, i] since it's 1
    
    # Backward substitution to solve Ux = y
    x = torch.zeros_like(y)
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x

# Create a right-hand side vector
b_3d_small = torch.ones(A_3d_small.shape[0])

# Solve the system
x_3d_small = solve_with_lu(A_3d_small, b_3d_small)

# Verify the solution
residual = torch.norm(A_3d_small @ x_3d_small - b_3d_small).item()
print(f"Solution residual: {residual:.2e}")

# Visualize the solution
plt.figure(figsize=(10, 6))
plt.plot(x_3d_small.numpy(), 'o-', markersize=8)
plt.grid(True, alpha=0.3)
plt.xlabel("Grid Point Index")
plt.ylabel("Solution Value")
plt.title("Solution of 3D Grid Problem")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Performance Comparison for Large Systems
# 
# Let's compare the performance of different methods for solving a large linear system:
# 
# 1. LU decomposition with our implementation
# 2. LU decomposition with NumPy/PyTorch
# 3. Direct solve using PyTorch's built-in solver

# %%
def compare_solving_methods(A, b):
    """Compare different methods for solving a linear system Ax = b."""
    n = A.shape[0]
    
    methods = []
    times = []
    residuals = []
    
    # Method 1: Our LU implementation
    start_time = time.time()
    x_our = solve_with_lu(A, b)
    times.append(time.time() - start_time)
    residuals.append(torch.norm(A @ x_our - b).item())
    methods.append("Our LU")
    
    # Method 2: NumPy's LU
    start_time = time.time()
    A_np = A.numpy()
    b_np = b.numpy()
    P, L, U = np.linalg.lu(A_np)
    # Forward and backward substitution
    y = np.linalg.solve(L, P @ b_np)
    x_np = np.linalg.solve(U, y)
    times.append(time.time() - start_time)
    residuals.append(np.linalg.norm(A_np @ x_np - b_np))
    methods.append("NumPy LU")
    
    # Method 3: PyTorch's direct solve
    start_time = time.time()
    x_torch = torch.linalg.solve(A, b)
    times.append(time.time() - start_time)
    residuals.append(torch.norm(A @ x_torch - b).item())
    methods.append("PyTorch Solve")
    
    # Method 4: PyTorch's LU solve
    start_time = time.time()
    LU, pivots = torch.linalg.lu_factor(A)
    x_torch_lu = torch.linalg.lu_solve(LU, pivots, b)
    times.append(time.time() - start_time)
    residuals.append(torch.norm(A @ x_torch_lu - b).item())
    methods.append("PyTorch LU")
    
    # Plot the results
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, times)
    plt.ylabel("Time (seconds)")
    plt.title("Solving Time Comparison")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, residuals)
    plt.ylabel("Residual ||Ax - b||")
    plt.title("Solution Accuracy Comparison")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print the comparison
    print("Performance Comparison:")
    for method, time_taken, residual in zip(methods, times, residuals):
        print(f"{method}: Time = {time_taken:.4f}s, Residual = {residual:.2e}")

# Create a right-hand side vector for the larger problem
b_3d = torch.ones(A_3d.shape[0])

# Compare solving methods
compare_solving_methods(A_3d, b_3d)

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we have explored LU decomposition, a fundamental technique in numerical linear algebra:
# 
# 1. We implemented LU decomposition from scratch and visualized the step-by-step process
# 2. We demonstrated the importance of pivoting for numerical stability
# 3. We compared our implementation with built-in functions
# 4. We showed how to use LU decomposition to solve linear systems
# 5. We examined a practical application with a 3D grid problem
# 
# LU decomposition has several advantages:
# 
# - It's efficient for solving multiple linear systems with the same coefficient matrix
# - It provides insight into the structure of the matrix
# - It can be used for calculating determinants and matrix inverses
# 
# However, it also has limitations:
# 
# - It requires O(n³) operations, which can be prohibitive for very large matrices
# - For sparse matrices, specialized sparse matrix techniques are often more efficient
# - Numerical stability requires pivoting, which adds complexity
# 
# In the next notebook, we'll explore more advanced aspects of LU decomposition and its applications.