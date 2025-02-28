# %% [markdown]
# # LU Decomposition: Applications
# 
# This notebook explores practical applications of LU decomposition in various domains, focusing on how this matrix factorization technique enables efficient computation for real-world problems.
# 
# We'll investigate the following applications:
# 
# 1. **Solving Systems of Linear Equations**
# 2. **Computing Matrix Inverse**
# 3. **Calculating Determinants**
# 4. **Circuit Analysis**
# 5. **Image Processing**
# 
# Each application demonstrates the computational advantages of LU decomposition and provides insights into its practical implementation.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import time
from PIL import Image
import requests
from io import BytesIO
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg
import matplotlib.patches as patches

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
# First, let's implement a reusable LU decomposition function and visualization helpers:

# %%
def lu_decomposition(A, pivot=True):
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
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
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
        if torch.abs(A_work[k, k]) < 1e-12:
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

def plot_matrix(matrix, title="Matrix", annotate=True, cmap=blue_cmap):
    """Plot a matrix as a heatmap with annotations."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.detach().cpu().numpy()
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

def forward_substitution(L, b):
    """Solve Ly = b for y using forward substitution."""
    n = L.shape[0]
    y = torch.zeros_like(b)
    
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        # L[i, i] is 1 so no need to divide
    
    return y

def backward_substitution(U, y):
    """Solve Ux = y for x using backward substitution."""
    n = U.shape[0]
    x = torch.zeros_like(y)
    
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x

def solve_with_lu(A, b, use_pivoting=True):
    """Solve a linear system Ax = b using LU decomposition."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float64)
        
    # Perform LU decomposition
    if use_pivoting:
        P, L, U = lu_decomposition(A, pivot=True)
        b_permuted = P @ b
    else:
        L, U = lu_decomposition(A, pivot=False)
        b_permuted = b
    
    # Forward substitution to solve Ly = b
    y = forward_substitution(L, b_permuted)
    
    # Backward substitution to solve Ux = y
    x = backward_substitution(U, y)
    
    return x

# %% [markdown]
# ## 1. Solving Systems of Linear Equations
# 
# One of the most important applications of LU decomposition is solving systems of linear equations efficiently. When we have multiple systems with the same coefficient matrix but different right-hand sides, LU decomposition is particularly advantageous.
# 
# $$A \cdot x_i = b_i \quad \text{for } i = 1, 2, \ldots, k$$
# 
# Let's demonstrate how LU decomposition helps in this scenario:

# %%
def solve_multiple_systems_example():
    """Demonstrate solving multiple linear systems with the same coefficient matrix."""
    # Create a matrix
    n = 100
    np.random.seed(42)
    A = np.random.rand(n, n)
    # Make it diagonally dominant for better conditioning
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    
    # Create multiple right-hand sides
    k = 5  # Number of systems
    B = np.random.rand(n, k)
    
    # Method 1: Solve each system separately using numpy.linalg.solve
    start_time = time.time()
    X_separate = np.zeros((n, k))
    for i in range(k):
        X_separate[:, i] = np.linalg.solve(A, B[:, i])
    separate_time = time.time() - start_time
    
    # Method 2: Use LU decomposition once, then solve multiple systems
    start_time = time.time()
    # Convert to PyTorch tensors
    A_torch = torch.tensor(A, dtype=torch.float64)
    B_torch = torch.tensor(B, dtype=torch.float64)
    
    # Perform LU decomposition once
    P, L, U = lu_decomposition(A_torch, pivot=True)
    
    # Solve multiple systems
    X_lu = torch.zeros((n, k), dtype=torch.float64)
    for i in range(k):
        b_permuted = P @ B_torch[:, i]
        y = forward_substitution(L, b_permuted)
        X_lu[:, i] = backward_substitution(U, y)
    lu_time = time.time() - start_time
    
    # Method 3: Use numpy's LU
    start_time = time.time()
    P_np, L_np, U_np = scipy.linalg.lu(A)
    X_lu_np = np.zeros((n, k))
    for i in range(k):
        y = scipy.linalg.solve_triangular(L_np, P_np @ B[:, i], lower=True)
        X_lu_np[:, i] = scipy.linalg.solve_triangular(U_np, y, lower=False)
    np_lu_time = time.time() - start_time
    
    # Method 4: Direct solve for all systems at once using numpy
    start_time = time.time()
    X_direct = np.linalg.solve(A, B)
    direct_time = time.time() - start_time
    
    # Calculate errors
    error_separate = np.linalg.norm(A @ X_separate - B) / np.linalg.norm(B)
    error_lu = np.linalg.norm(A @ X_lu.numpy() - B) / np.linalg.norm(B)
    error_lu_np = np.linalg.norm(A @ X_lu_np - B) / np.linalg.norm(B)
    error_direct = np.linalg.norm(A @ X_direct - B) / np.linalg.norm(B)
    
    # Compare timing and accuracy
    methods = ["Separate Solves", "Our LU", "NumPy LU", "Direct Solve"]
    times = [separate_time, lu_time, np_lu_time, direct_time]
    errors = [error_separate, error_lu, error_lu_np, error_direct]
    
    # Plot the results
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, times)
    plt.ylabel("Time (seconds)")
    plt.title(f"Solving {k} Linear Systems (Matrix Size: {n}×{n})")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Annotate with speedup
    for i, t in enumerate(times):
        plt.text(i, t + 0.01 * max(times), f"{t:.4f}s", 
                 ha='center', va='bottom', fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, errors)
    plt.ylabel("Relative Error")
    plt.title("Solution Accuracy")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Annotate with error values
    for i, e in enumerate(errors):
        plt.text(i, e * 1.1, f"{e:.2e}", 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Performance Comparison for Solving Multiple Linear Systems:")
    print("-" * 70)
    print(f"{'Method':<20} {'Time (s)':<15} {'Speedup':<15} {'Relative Error':<15}")
    print("-" * 70)
    
    baseline = separate_time
    for method, t, e in zip(methods, times, errors):
        speedup = baseline / t if t > 0 else float('inf')
        print(f"{method:<20} {t:<15.4f} {speedup:<15.2f} {e:<15.2e}")
    
    # Return the fastest method's solution
    return X_lu.numpy()

# Demonstrate solving multiple systems
X_solution = solve_multiple_systems_example()

# %% [markdown]
# The graph above demonstrates a key advantage of LU decomposition: once we compute the factorization, we can solve multiple systems with different right-hand sides much more efficiently.
# 
# Let's dive deeper into how the performance advantage of LU decomposition scales with the number of systems:

# %%
def lu_performance_scaling():
    """Explore how LU performance scales with the number of systems."""
    # Create a matrix
    n = 100
    np.random.seed(42)
    A = np.random.rand(n, n)
    # Make it diagonally dominant for better conditioning
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    
    # Try different numbers of systems
    system_counts = [1, 2, 5, 10, 20, 50]
    separate_times = []
    lu_times = []
    
    for k in system_counts:
        # Create k right-hand sides
        B = np.random.rand(n, k)
        
        # Method 1: Solve each system separately
        start_time = time.time()
        X_separate = np.zeros((n, k))
        for i in range(k):
            X_separate[:, i] = np.linalg.solve(A, B[:, i])
        separate_times.append(time.time() - start_time)
        
        # Method 2: Use LU decomposition
        start_time = time.time()
        P_np, L_np, U_np = scipy.linalg.lu(A)
        X_lu = np.zeros((n, k))
        for i in range(k):
            y = scipy.linalg.solve_triangular(L_np, P_np @ B[:, i], lower=True)
            X_lu[:, i] = scipy.linalg.solve_triangular(U_np, y, lower=False)
        lu_times.append(time.time() - start_time)
    
    # Plot scaling behavior
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(system_counts, separate_times, 'o-', label='Direct Solve', linewidth=2)
    plt.plot(system_counts, lu_times, 's-', label='LU Decomposition', linewidth=2)
    plt.xlabel("Number of Systems")
    plt.ylabel("Time (seconds)")
    plt.title("Scaling with Number of Systems")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    speedups = [s/l for s, l in zip(separate_times, lu_times)]
    plt.plot(system_counts, speedups, 'd-', color='green', linewidth=2)
    plt.xlabel("Number of Systems")
    plt.ylabel("Speedup Factor")
    plt.title("LU Speedup vs. Direct Solve")
    plt.grid(True, alpha=0.3)
    
    # Add trendline
    z = np.polyfit(system_counts, speedups, 1)
    p = np.poly1d(z)
    plt.plot(system_counts, p(system_counts), 'r--', alpha=0.7, 
             label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("LU Decomposition Scaling Analysis:")
    print("-" * 70)
    print(f"{'Number of Systems':<20} {'Direct Time (s)':<20} {'LU Time (s)':<20} {'Speedup':<15}")
    print("-" * 70)
    
    for k, s, l, speedup in zip(system_counts, separate_times, lu_times, speedups):
        print(f"{k:<20} {s:<20.4f} {l:<20.4f} {speedup:<15.2f}")
    
    # Calculate the breakeven point
    # At what number of systems does LU start to be more efficient?
    # For a simple model: Direct = a*k, LU = b + c*k
    # Breakeven: a*k = b + c*k => k = b/(a-c)
    
    if len(system_counts) >= 2:
        # Estimate coefficients
        a = separate_times[1] / system_counts[1]  # Direct solve cost per system
        b = lu_times[0] - lu_times[1]/system_counts[1] * system_counts[0]  # LU fixed cost
        c = lu_times[1]/system_counts[1]  # LU marginal cost per system
        
        if a > c:
            breakeven = b / (a - c)
            print(f"\nEstimated breakeven point: {breakeven:.2f} systems")
            print(f"For {n}x{n} matrices, LU is more efficient when solving more than {int(np.ceil(breakeven))} systems")
        else:
            print("\nLU appears to be more efficient for any number of systems with this matrix size")

# Analyze how LU performance scales with the number of systems
lu_performance_scaling()

# %% [markdown]
# ## 2. Computing Matrix Inverse
# 
# Another important application of LU decomposition is computing the inverse of a matrix. The inverse A⁻¹ of a square matrix A satisfies:
# 
# $$A \cdot A^{-1} = A^{-1} \cdot A = I$$
# 
# We can compute the inverse by:
# 1. Performing LU decomposition of A
# 2. Solving n different systems, one for each column of the identity matrix
# 
# Let's implement and visualize this process:

# %%
def compute_inverse_with_lu(A):
    """Compute the inverse of matrix A using LU decomposition."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    
    # Perform LU decomposition with pivoting
    P, L, U = lu_decomposition(A, pivot=True)
    
    # Initialize the inverse matrix
    A_inv = torch.zeros((n, n), dtype=A.dtype)
    
    # For each column of the identity matrix
    for j in range(n):
        # Create the j-th column of the identity matrix
        e_j = torch.zeros(n, dtype=A.dtype)
        e_j[j] = 1.0
        
        # Solve the system A⁻¹[:,j] = x where Ax = e_j
        b_permuted = P @ e_j
        y = forward_substitution(L, b_permuted)
        A_inv[:, j] = backward_substitution(U, y)
    
    return A_inv

def visualize_matrix_inverse():
    """Visualize a matrix and its inverse."""
    # Create a well-conditioned 4x4 matrix
    np.random.seed(42)
    A = np.random.rand(4, 4)
    # Make it diagonally dominant
    for i in range(4):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    
    # Compute the inverse using LU decomposition
    A_inv_lu = compute_inverse_with_lu(A)
    
    # Compare with NumPy's inverse
    A_inv_np = np.linalg.inv(A)
    
    # Check that A·A⁻¹ ≈ I
    I_lu = A @ A_inv_lu.numpy()
    I_np = A @ A_inv_np
    
    # Visualize
    plt.figure(figsize=(15, 12))
    
    # Original matrix
    plt.subplot(2, 3, 1)
    sns.heatmap(A, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Original Matrix A")
    
    # LU-based inverse
    plt.subplot(2, 3, 2)
    sns.heatmap(A_inv_lu.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Inverse A⁻¹ (LU)")
    
    # Product A·A⁻¹
    plt.subplot(2, 3, 3)
    sns.heatmap(I_lu, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("A·A⁻¹ (LU) ≈ I")
    
    # NumPy inverse
    plt.subplot(2, 3, 5)
    sns.heatmap(A_inv_np, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Inverse A⁻¹ (NumPy)")
    
    # Product A·A⁻¹
    plt.subplot(2, 3, 6)
    sns.heatmap(I_np, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("A·A⁻¹ (NumPy) ≈ I")
    
    plt.tight_layout()
    plt.show()
    
    # Calculate error
    lu_error = np.linalg.norm(I_lu - np.eye(4), 'fro') / 4
    np_error = np.linalg.norm(I_np - np.eye(4), 'fro') / 4
    
    print("Matrix Inverse Computation:")
    print(f"LU Decomposition Error: {lu_error:.2e}")
    print(f"NumPy Error: {np_error:.2e}")
    
    return A, A_inv_lu.numpy()

# Visualize matrix inverse computation
A_original, A_inverse = visualize_matrix_inverse()

# %% [markdown]
# ## 3. Calculating Determinants
# 
# LU decomposition provides an efficient way to calculate the determinant of a matrix. If $A = LU$ (without pivoting), then:
# 
# $$\det(A) = \det(L) \cdot \det(U)$$
# 
# Since $L$ is lower triangular with ones on the diagonal, $\det(L) = 1$. And for a triangular matrix, the determinant is the product of the diagonal elements. So:
# 
# $$\det(A) = \det(U) = \prod_{i=1}^{n} U_{ii}$$
# 
# When pivoting is used, we have $PA = LU$, so:
# 
# $$\det(A) = \det(P)^{-1} \cdot \det(L) \cdot \det(U)$$
# 
# where $\det(P) = \pm 1$ depending on whether an even or odd number of row exchanges were performed.

# %%
def calculate_determinant_with_lu(A):
    """Calculate determinant using LU decomposition."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    # Perform LU decomposition with pivoting
    P, L, U = lu_decomposition(A, pivot=True)
    
    # Calculate determinant of U (product of diagonal elements)
    det_U = torch.prod(torch.diag(U))
    
    # Calculate determinant of P
    # For a permutation matrix, det(P) = (-1)^s where s is the number of row swaps
    # Instead of counting swaps, we can compute det(P) directly
    det_P = torch.linalg.det(P)
    
    # Calculate determinant of A
    det_A = det_U / det_P  # det(P)^-1 * det(L) * det(U), where det(L) = 1
    
    return det_A.item()

def compare_determinant_methods():
    """Compare different methods for calculating determinants."""
    # Create matrices of different sizes
    sizes = [5, 10, 20, 50, 100, 200]
    lu_times = []
    direct_times = []
    errors = []
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.rand(n, n)
        
        # LU method
        start_time = time.time()
        det_lu = calculate_determinant_with_lu(A)
        lu_times.append(time.time() - start_time)
        
        # Direct method
        start_time = time.time()
        det_direct = np.linalg.det(A)
        direct_times.append(time.time() - start_time)
        
        # Calculate relative error
        if abs(det_direct) > 1e-10:
            error = abs((det_lu - det_direct) / det_direct)
        else:
            error = abs(det_lu - det_direct)
        errors.append(error)
    
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(sizes, lu_times, 'o-', label='LU Method', linewidth=2)
    plt.plot(sizes, direct_times, 's-', label='Direct Method', linewidth=2)
    plt.xlabel("Matrix Size")
    plt.ylabel("Time (seconds)")
    plt.title("Determinant Calculation Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 3, 2)
    speedups = [d/l for d, l in zip(direct_times, lu_times)]
    plt.plot(sizes, speedups, 'd-', color='green', linewidth=2)
    plt.xlabel("Matrix Size")
    plt.ylabel("Speedup Factor")
    plt.title("LU Speedup vs. Direct Method")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.semilogy(sizes, errors, 'o-', color='red', linewidth=2)
    plt.xlabel("Matrix Size")
    plt.ylabel("Relative Error")
    plt.title("Accuracy of LU Method")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Determinant Calculation Comparison:")
    print("-" * 70)
    print(f"{'Matrix Size':<15} {'LU Time (s)':<15} {'Direct Time (s)':<15} {'Speedup':<15} {'Relative Error':<15}")
    print("-" * 70)
    
    for n, lt, dt, speedup, error in zip(sizes, lu_times, direct_times, speedups, errors):
        print(f"{n:<15} {lt:<15.6f} {dt:<15.6f} {speedup:<15.2f} {error:<15.2e}")
    
    # Create a special example to demonstrate
    n = 4
    A_demo = np.array([
        [2, 1, 3, 4],
        [4, 5, 6, 7],
        [8, 9, 1, 2],
        [3, 4, 5, 9]
    ])
    
    # Calculate determinant
    P, L, U = lu_decomposition(torch.tensor(A_demo, dtype=torch.float64), pivot=True)
    det_A = np.linalg.det(A_demo)
    det_U = np.prod(np.diag(U.numpy()))
    det_P = np.linalg.det(P.numpy())
    
    # Visualize the process
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    sns.heatmap(A_demo, annot=True, fmt=".0f", cmap=blue_cmap, linewidths=1)
    plt.title(f"Matrix A\ndet(A) = {det_A:.2f}")
    
    plt.subplot(1, 4, 2)
    sns.heatmap(P.numpy(), annot=True, fmt=".0f", cmap=blue_cmap, linewidths=1)
    plt.title(f"Permutation P\ndet(P) = {det_P:.0f}")
    
    plt.subplot(1, 4, 3)
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title("Lower Triangular L\ndet(L) = 1")
    
    plt.subplot(1, 4, 4)
    sns.heatmap(U.numpy(), annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1)
    plt.title(f"Upper Triangular U\ndet(U) = {det_U:.2f}")
    
    plt.tight_layout()
    plt.show()
    
    print("\nExample Calculation:")
    print(f"Matrix A: det(A) = {det_A:.2f}")
    print(f"LU Decomposition: det(A) = det(P)^-1 × det(L) × det(U) = {1/det_P:.0f} × 1 × {det_U:.2f} = {det_U/det_P:.2f}")

# Compare determinant calculation methods
compare_determinant_methods()

# %% [markdown]
# ## 4. Circuit Analysis
# 
# LU decomposition can be applied to analyze electrical circuits using nodal analysis or mesh analysis, which result in systems of linear equations.
# 
# Let's consider a simple resistor network and use LU decomposition to solve for the node voltages:

# %%
def analyze_resistor_network():
    """Analyze a resistor network using LU decomposition."""
    # Create a simple resistor network
    # We'll use a 3-node circuit with resistors between nodes
    
    # Define resistor values in ohms
    R12 = 10.0  # Resistor between nodes 1 and 2
    R23 = 20.0  # Resistor between nodes 2 and 3
    R13 = 30.0  # Resistor between nodes 1 and 3
    R10 = 5.0   # Resistor between node 1 and ground
    R20 = 15.0  # Resistor between node 2 and ground
    R30 = 25.0  # Resistor between node 3 and ground
    
    # Define current sources in amperes
    I1 = 1.0    # Current into node 1
    I2 = 0.0    # Current into node 2
    I3 = -0.5   # Current into node 3 (negative means out of the node)
    
    # Conductance matrix (G = 1/R)
    G = np.zeros((3, 3))
    
    # Fill diagonal elements with the sum of conductances connected to the node
    G[0, 0] = 1/R12 + 1/R13 + 1/R10
    G[1, 1] = 1/R12 + 1/R23 + 1/R20
    G[2, 2] = 1/R13 + 1/R23 + 1/R30
    
    # Fill off-diagonal elements with the negative conductance between nodes
    G[0, 1] = G[1, 0] = -1/R12
    G[0, 2] = G[2, 0] = -1/R13
    G[1, 2] = G[2, 1] = -1/R23
    
    # Current vector
    I = np.array([I1, I2, I3])
    
    # Solve for node voltages using LU decomposition
    V = solve_with_lu(G, I)
    
    # For comparison, solve using numpy
    V_np = np.linalg.solve(G, I)
    
    # Visualize the circuit
    plt.figure(figsize=(15, 7))
    
    # Draw circuit diagram
    plt.subplot(1, 2, 1)
    plt.axis('equal')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    
    # Draw nodes
    node_positions = {
        0: (5, 0),    # Ground
        1: (2, 7),    # Node 1
        2: (5, 7),    # Node 2
        3: (8, 7)     # Node 3
    }
    
    # Draw ground
    plt.plot([4, 6], [0, 0], 'k-', linewidth=2)
    plt.plot([4.2, 5.8], [0.2, 0.2], 'k-', linewidth=2)
    plt.plot([4.4, 5.6], [0.4, 0.4], 'k-', linewidth=2)
    
    # Draw nodes
    for node, pos in node_positions.items():
        if node == 0:  # Ground
            continue
        plt.plot(pos[0], pos[1], 'ko', markersize=10)
        plt.text(pos[0], pos[1] + 0.5, f"Node {node}\n{V.numpy()[node-1]:.2f}V", 
                 ha='center', va='bottom', fontsize=12)
    
    # Draw resistors
    def draw_resistor(pos1, pos2, value, name, offset=(0, 0)):
        # Draw line for resistor
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'k-', linewidth=1.5)
        
        # Draw resistor symbol
        resistor = patches.Rectangle((mid_x-0.5, mid_y-0.3), 1, 0.6, 
                                    linewidth=1.5, edgecolor='k', facecolor='white')
        plt.gca().add_patch(resistor)
        
        # Add value label
        plt.text(mid_x + offset[0], mid_y + offset[1], f"{name}={value}Ω", 
                ha='center', va='bottom', fontsize=10)
    
    # Draw resistors between nodes
    draw_resistor(node_positions[1], node_positions[2], R12, "R12", (0, 0.5))
    draw_resistor(node_positions[2], node_positions[3], R23, "R23", (0, 0.5))
    draw_resistor(node_positions[1], node_positions[3], R13, "R13", (0, -0.7))
    
    # Draw resistors to ground
    draw_resistor(node_positions[1], (node_positions[1][0], 0), R10, "R10", (-0.7, 0))
    draw_resistor(node_positions[2], (node_positions[2][0], 0), R20, "R20", (0, 0))
    draw_resistor(node_positions[3], (node_positions[3][0], 0), R30, "R30", (0.7, 0))
    
    # Draw current sources
    def draw_current_source(pos, value, name, direction='down'):
        if direction == 'down':
            plt.arrow(pos[0], pos[1] + 1.5, 0, -0.8, head_width=0.2, head_length=0.4, 
                     fc='blue', ec='blue', linewidth=2)
            plt.text(pos[0] + 0.3, pos[1] + 1, f"{name}={value}A", 
                    ha='left', va='center', fontsize=10)
        elif direction == 'up':
            plt.arrow(pos[0], pos[1] - 1.5, 0, 0.8, head_width=0.2, head_length=0.4, 
                     fc='blue', ec='blue', linewidth=2)
            plt.text(pos[0] + 0.3, pos[1] - 1, f"{name}={-value}A", 
                    ha='left', va='center', fontsize=10)
    
    # Draw current sources
    draw_current_source(node_positions[1], I1, "I1", 'down')
    draw_current_source(node_positions[3], I3, "I3", 'up')
    
    plt.axis('off')
    plt.title("Resistor Network")
    
    # Visualize the conductance matrix and solution
    plt.subplot(1, 2, 2)
    sns.heatmap(G, annot=True, fmt=".3f", cmap=blue_cmap, linewidths=1, 
               xticklabels=["Node 1", "Node 2", "Node 3"],
               yticklabels=["Node 1", "Node 2", "Node 3"])
    plt.title("Conductance Matrix G")
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("Circuit Analysis Results:")
    print("-" * 50)
    print(f"{'Node':<10} {'Voltage (LU)':<15} {'Voltage (NumPy)':<15} {'Difference':<15}")
    print("-" * 50)
    
    for i in range(3):
        print(f"{i+1:<10} {V.numpy()[i]:<15.6f} {V_np[i]:<15.6f} {abs(V.numpy()[i] - V_np[i]):<15.2e}")
    
    # Calculate currents through each resistor
    currents = {}
    currents["I_R12"] = (V.numpy()[0] - V.numpy()[1]) / R12
    currents["I_R23"] = (V.numpy()[1] - V.numpy()[2]) / R23
    currents["I_R13"] = (V.numpy()[0] - V.numpy()[2]) / R13
    currents["I_R10"] = V.numpy()[0] / R10
    currents["I_R20"] = V.numpy()[1] / R20
    currents["I_R30"] = V.numpy()[2] / R30
    
    print("\nCurrent through each resistor:")
    for name, current in currents.items():
        print(f"{name}: {current:.4f} A")
    
    # Verify Kirchhoff's Current Law (KCL) at each node
    kcl_1 = I1 - currents["I_R12"] - currents["I_R13"] - currents["I_R10"]
    kcl_2 = I2 + currents["I_R12"] - currents["I_R23"] - currents["I_R20"]
    kcl_3 = I3 + currents["I_R13"] + currents["I_R23"] - currents["I_R30"]
    
    print("\nKirchhoff's Current Law Verification:")
    print(f"Node 1: {kcl_1:.2e} (should be close to zero)")
    print(f"Node 2: {kcl_2:.2e} (should be close to zero)")
    print(f"Node 3: {kcl_3:.2e} (should be close to zero)")
    
    return G, I, V.numpy()

# Analyze a resistor network
G_matrix, I_vector, V_solution = analyze_resistor_network()

# %% [markdown]
# ## 5. Image Processing
# 
# LU decomposition can also be applied to image processing tasks like image compression and reconstruction. Let's demonstrate how it can be used to solve a simple image denoising problem:

# %%
def image_denoising_with_lu():
    """Demonstrate image denoising using LU decomposition to solve a linear system."""
    # Load a small grayscale image or create a synthetic one
    try:
        # Try to load an image
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Claude_Shannon.jpg/220px-Claude_Shannon.jpg"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('L')
        img = img.resize((64, 64))  # Resize for faster computation
        img_array = np.array(img) / 255.0
    except:
        # Create a synthetic image if loading fails
        n = 64
        x, y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        # Create a circle
        circle = (x**2 + y**2 < 0.5**2).astype(float)
        # Add some patterns
        img_array = circle + 0.5 * np.sin(5 * x) * np.cos(5 * y) * circle
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    
    # Add noise to the image
    np.random.seed(42)
    noise_level = 0.2
    noisy_array = img_array + noise_level * np.random.randn(*img_array.shape)
    noisy_array = np.clip(noisy_array, 0, 1)
    
    # Create regularization matrix for Tikhonov regularization
    # This will encourage smoothness in the solution
    n = img_array.size
    side_length = int(np.sqrt(n))
    
    # We'll use a sparse representation for efficiency
    # Create a Laplacian operator in matrix form
    row_indices = []
    col_indices = []
    values = []
    
    for i in range(side_length):
        for j in range(side_length):
            idx = i * side_length + j
            
            # Add diagonal element
            row_indices.append(idx)
            col_indices.append(idx)
            values.append(4.0)  # 4 for central point
            
            # Add neighbors (if they exist)
            neighbors = []
            if i > 0:
                neighbors.append((i-1) * side_length + j)  # Up
            if i < side_length-1:
                neighbors.append((i+1) * side_length + j)  # Down
            if j > 0:
                neighbors.append(i * side_length + (j-1))  # Left
            if j < side_length-1:
                neighbors.append(i * side_length + (j+1))  # Right
            
            for neighbor_idx in neighbors:
                row_indices.append(idx)
                col_indices.append(neighbor_idx)
                values.append(-1.0)  # -1 for neighbors
    
    # Create the sparse Laplacian matrix
    L = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(n, n))
    L = L.tocsr()
    
    # Set up the linear system for Tikhonov regularization
    # (A^T A + lambda L^T L)x = A^T b
    # Where A is the identity matrix (for denoising), lambda is the regularization parameter,
    # L is the Laplacian, b is the noisy image, and x is the denoised image
    
    lambda_reg = 0.1  # Regularization parameter
    
    # A^T A + lambda L^T L
    regularized_matrix = sparse.eye(n) + lambda_reg * (L.T @ L)
    
    # Convert to numpy for our solver
    reg_matrix_dense = regularized_matrix.toarray()
    
    # Solve the linear system
    start_time = time.time()
    denoised_flat = solve_with_lu(reg_matrix_dense, noisy_array.flatten())
    lu_time = time.time() - start_time
    
    # Reshape the result
    denoised_array = denoised_flat.numpy().reshape(img_array.shape)
    
    # For comparison, also solve using a sparse direct solver
    start_time = time.time()
    denoised_flat_sparse = spalg.spsolve(regularized_matrix, noisy_array.flatten())
    sparse_time = time.time() - start_time
    denoised_array_sparse = denoised_flat_sparse.reshape(img_array.shape)
    
    # Visualize the results
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(noisy_array, cmap='gray')
    plt.title(f"Noisy Image\nPSNR: {calculate_psnr(img_array, noisy_array):.2f} dB")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(denoised_array, cmap='gray')
    plt.title(f"Denoised Image (LU)\nPSNR: {calculate_psnr(img_array, denoised_array):.2f} dB\nTime: {lu_time:.2f}s")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(denoised_array_sparse, cmap='gray')
    plt.title(f"Denoised Image (Sparse)\nPSNR: {calculate_psnr(img_array, denoised_array_sparse):.2f} dB\nTime: {sparse_time:.2f}s")
    plt.axis('off')
    
    # Visualize a small part of the regularization matrix and Laplacian
    plt.subplot(2, 3, 4)
    display_size = min(20, side_length)  # Show at most 20x20 section
    plt.imshow(reg_matrix_dense[:display_size, :display_size], cmap=blue_cmap)
    plt.title("Regularization Matrix\n(Top-Left Corner)")
    plt.colorbar()
    
    plt.subplot(2, 3, 5)
    plt.spy(L[:100, :100], markersize=2)
    plt.title("Laplacian Matrix Structure\n(Top-Left Corner)")
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Image Denoising Results:")
    print(f"Image size: {img_array.shape[0]}×{img_array.shape[1]} ({n} pixels)")
    print(f"Noise level: {noise_level}")
    print(f"Regularization parameter: {lambda_reg}")
    print("-" * 70)
    print(f"{'Method':<15} {'PSNR (dB)':<15} {'Time (s)':<15} {'Improvement (dB)':<20}")
    print("-" * 70)
    
    noisy_psnr = calculate_psnr(img_array, noisy_array)
    lu_psnr = calculate_psnr(img_array, denoised_array)
    sparse_psnr = calculate_psnr(img_array, denoised_array_sparse)
    
    print(f"{'Noisy Image':<15} {noisy_psnr:<15.2f} {'-':<15} {'-':<20}")
    print(f"{'LU Denoising':<15} {lu_psnr:<15.2f} {lu_time:<15.2f} {lu_psnr - noisy_psnr:<20.2f}")
    print(f"{'Sparse Solver':<15} {sparse_psnr:<15.2f} {sparse_time:<15.2f} {sparse_psnr - noisy_psnr:<20.2f}")
    
    return img_array, noisy_array, denoised_array

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio between two images."""
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Demonstrate image denoising
original_img, noisy_img, denoised_img = image_denoising_with_lu()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored a variety of applications of LU decomposition:
# 
# 1. **Solving Systems of Linear Equations**: LU decomposition is particularly efficient when solving multiple systems with the same coefficient matrix, showing significant speedup as the number of systems increases.
# 
# 2. **Computing Matrix Inverse**: We implemented inverse calculation using LU decomposition, demonstrating how this approach works by solving n linear systems.
# 
# 3. **Calculating Determinants**: LU decomposition provides an efficient way to compute determinants, which becomes increasingly advantageous for larger matrices.
# 
# 4. **Circuit Analysis**: We applied LU decomposition to solve for node voltages in a resistor network, demonstrating its usefulness in electrical engineering.
# 
# 5. **Image Processing**: LU decomposition can be used in image denoising and restoration problems, though sparse solvers may be more efficient for larger images.
# 
# LU decomposition is a fundamental technique in numerical linear algebra with widespread applications in science and engineering. Its versatility makes it a valuable tool in any computational mathematician's or engineer's toolkit.
# 
# Key advantages of LU decomposition:
# 
# - Computational efficiency for multiple related problems
# - Numerical stability (with pivoting)
# - Ability to compute determinants and inverses efficiently
# - Applicability to a wide range of practical problems
# 
# The main limitation is the O(n³) complexity, which can be prohibitive for very large systems. For such cases, iterative methods or specialized sparse solvers may be more appropriate.