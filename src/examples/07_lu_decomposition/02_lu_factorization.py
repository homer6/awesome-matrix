# %% [markdown]
# # LU Decomposition: Factorization Methods
# 
# In this notebook, we explore different methods for computing LU decomposition and their implementation details. We examine the mathematical foundations, algorithms, and potential optimizations for LU factorization.
# 
# We'll focus on:
# 
# 1. **Mathematical Theory of LU Factorization**
# 2. **Variants of LU Decomposition (LDU, PLU, etc.)**
# 3. **Optimized Implementations**
# 4. **Stability and Accuracy Considerations**
# 
# This notebook bridges the introduction to LU decomposition and its practical applications, providing deeper insight into the factorization process itself.

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

# Helper functions for visualization
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

# %% [markdown]
# ## 1. Mathematical Theory of LU Factorization
# 
# LU decomposition expresses a matrix $A$ as the product of a lower triangular matrix $L$ and an upper triangular matrix $U$:
# 
# $$A = LU$$
# 
# This factorization is closely related to Gaussian elimination, which transforms a matrix into row echelon form through a series of row operations.
# 
# ### Conditions for Existence
# 
# A square matrix $A$ admits an LU factorization without pivoting if and only if all its leading principal minors are non-zero. In other words, the determinants of the sub-matrices formed by the first $k$ rows and columns must be non-zero for $k = 1, 2, \ldots, n-1$.
# 
# Let's demonstrate these conditions with examples:

# %%
def check_lu_existence(A):
    """Check if LU factorization exists without pivoting."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    exists = True
    minor_values = []
    
    for k in range(1, n):
        # Extract leading principal minor
        minor = A[:k, :k]
        # Calculate determinant
        det = torch.linalg.det(minor).item()
        minor_values.append(det)
        
        if abs(det) < 1e-10:
            exists = False
            break
    
    return exists, minor_values

def demonstrate_lu_existence():
    """Demonstrate matrices with and without LU factorization."""
    # Matrix that has LU factorization
    A_good = torch.tensor([
        [2.0, 1.0, 1.0],
        [4.0, 3.0, 3.0],
        [8.0, 7.0, 9.0]
    ])
    
    # Matrix that requires pivoting (first leading minor is zero)
    A_bad = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0]
    ])
    
    # Check existence
    exists_good, minors_good = check_lu_existence(A_good)
    exists_bad, minors_bad = check_lu_existence(A_bad)
    
    # Display the matrices
    plot_matrix(A_good, "Matrix with LU Factorization")
    plot_matrix(A_bad, "Matrix without LU Factorization (Needs Pivoting)")
    
    # Show leading principal minors
    print("Leading Principal Minors for Matrix with LU Factorization:")
    for i, det in enumerate(minors_good):
        print(f"Order {i+1}: {det:.6f}")
    
    print("\nLeading Principal Minors for Matrix without LU Factorization:")
    for i, det in enumerate(minors_bad):
        print(f"Order {i+1}: {det:.6f}")
    
    # Try to perform LU decomposition
    try:
        L_good, U_good = scipy.linalg.lu_factor(A_good.numpy())
        print("\nLU factorization successful for first matrix")
    except np.linalg.LinAlgError:
        print("\nLU factorization failed for first matrix")
    
    try:
        L_bad, U_bad = scipy.linalg.lu_factor(A_bad.numpy())
        print("LU factorization successful for second matrix (with pivoting)")
    except np.linalg.LinAlgError:
        print("LU factorization failed for second matrix")
    
    return A_good, A_bad

# Demonstrate existence conditions for LU factorization
A_good, A_bad = demonstrate_lu_existence()

# %% [markdown]
# ## 2. Variants of LU Decomposition
# 
# There are several variants of LU decomposition, each with specific properties and applications:
# 
# ### 2.1 LDU Decomposition
# 
# LDU decomposition expresses a matrix as $A = LDU$ where:
# - $L$ is a lower triangular matrix with ones on the diagonal
# - $D$ is a diagonal matrix
# - $U$ is an upper triangular matrix with ones on the diagonal
# 
# This is a more refined form of the LU decomposition that separates the diagonal elements.

# %%
def ldu_decomposition(A):
    """
    Compute LDU decomposition of a matrix A.
    
    Args:
        A: Input matrix as a PyTorch tensor
    
    Returns:
        L: Lower triangular matrix with ones on diagonal
        D: Diagonal matrix
        U: Upper triangular matrix with ones on diagonal
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    
    # Perform LU decomposition first
    P, L, U = scipy.linalg.lu(A.numpy())
    
    # Extract diagonal of U
    D = torch.diag(torch.diag(torch.tensor(U, dtype=torch.float64)))
    
    # Create U with ones on diagonal
    U_normalized = torch.tensor(U, dtype=torch.float64) @ torch.linalg.inv(D)
    
    return torch.tensor(L, dtype=torch.float64), D, U_normalized

def demonstrate_ldu():
    """Demonstrate LDU decomposition."""
    # Create a simple matrix
    A = torch.tensor([
        [4.0, 3.0, 2.0],
        [3.0, 5.0, 1.0],
        [2.0, 1.0, 6.0]
    ])
    
    # Compute LDU decomposition
    L, D, U = ldu_decomposition(A)
    
    # Display the matrices
    plot_matrix(A, "Original Matrix A")
    plot_matrix(L, "Lower Triangular Matrix L")
    plot_matrix(D, "Diagonal Matrix D")
    plot_matrix(U, "Upper Triangular Matrix U")
    
    # Verify the decomposition
    LDU = L @ D @ U
    plot_matrix(LDU, "Reconstructed Matrix LDU")
    
    # Calculate reconstruction error
    error = torch.norm(A - LDU).item()
    print(f"Reconstruction error: {error:.2e}")
    
    # Check that L and U have ones on their diagonals
    print("\nDiagonal of L:", torch.diag(L).numpy())
    print("Diagonal of U:", torch.diag(U).numpy())
    
    return A, L, D, U

# Demonstrate LDU decomposition
A_ldu, L_ldu, D_ldu, U_ldu = demonstrate_ldu()

# %% [markdown]
# ### 2.2 PLU Decomposition (with Partial Pivoting)
# 
# PLU decomposition incorporates row permutations to handle matrices that don't admit a direct LU factorization. It expresses a matrix as:
# 
# $$PA = LU$$
# 
# where $P$ is a permutation matrix that reorders the rows of $A$.
# 
# This variant is more robust and is implemented in most numerical libraries. Let's implement and demonstrate this pivoting process:

# %%
def lu_with_partial_pivoting(A):
    """
    Perform LU decomposition with partial pivoting.
    
    Args:
        A: Input matrix as a PyTorch tensor
    
    Returns:
        P: Permutation matrix
        L: Lower triangular matrix
        U: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    
    # Make a copy to avoid modifying the original matrix
    A_work = A.clone()
    
    # Initialize L as identity matrix
    L = torch.eye(n, dtype=A.dtype)
    
    # Initialize permutation matrix
    P = torch.eye(n, dtype=A.dtype)
    
    # Perform Gaussian elimination
    for k in range(n-1):  # Loop through each column
        # Find the index of the maximum absolute value in the current column (from k to n)
        max_idx = torch.argmax(torch.abs(A_work[k:, k])) + k
        
        # If the max is not at the current row, swap rows
        if max_idx != k:
            # Swap rows in A_work
            A_work[[k, max_idx], :] = A_work[[max_idx, k], :]
            
            # Swap rows in P
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
    
    return P, L, U

def demonstrate_plu():
    """Demonstrate PLU decomposition."""
    # Use the matrix that requires pivoting
    A = torch.tensor([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0]
    ])
    
    # Compute PLU decomposition
    P, L, U = lu_with_partial_pivoting(A)
    
    # Display the matrices
    plot_matrix(A, "Original Matrix A")
    plot_matrix(P, "Permutation Matrix P")
    plot_matrix(L, "Lower Triangular Matrix L")
    plot_matrix(U, "Upper Triangular Matrix U")
    
    # Verify that PA = LU
    PA = P @ A
    LU = L @ U
    
    plot_matrix(PA, "PA")
    plot_matrix(LU, "LU")
    
    # Calculate reconstruction error
    error = torch.norm(PA - LU).item()
    print(f"Reconstruction error: {error:.2e}")
    
    # Also try with scipy for comparison
    P_scipy, L_scipy, U_scipy = scipy.linalg.lu(A.numpy())
    
    # Convert to PyTorch tensors
    P_scipy = torch.tensor(P_scipy, dtype=torch.float64)
    L_scipy = torch.tensor(L_scipy, dtype=torch.float64)
    U_scipy = torch.tensor(U_scipy, dtype=torch.float64)
    
    # Verify scipy's result
    PA_scipy = P_scipy.T @ A
    LU_scipy = L_scipy @ U_scipy
    
    error_scipy = torch.norm(PA_scipy - LU_scipy).item()
    print(f"SciPy reconstruction error: {error_scipy:.2e}")
    
    return A, P, L, U

# Demonstrate PLU decomposition
A_plu, P_plu, L_plu, U_plu = demonstrate_plu()

# %% [markdown]
# ### 2.3 LU Factorization of Rectangular Matrices
# 
# While we often think of LU decomposition for square matrices, it can be extended to rectangular matrices as well. For an $m \times n$ matrix with $m \geq n$, the decomposition is:
# 
# $$A = LU$$
# 
# where $L$ is an $m \times n$ lower triangular matrix, and $U$ is an $n \times n$ upper triangular matrix.

# %%
def lu_rectangular(A):
    """
    Compute LU decomposition for a rectangular matrix.
    
    Args:
        A: Input rectangular matrix (m x n) with m >= n
        
    Returns:
        L: m x n lower triangular matrix
        U: n x n upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    assert m >= n, "Matrix must have at least as many rows as columns"
    
    # Use existing implementation with partial pivoting
    P, L_full, U = lu_with_partial_pivoting(A[:n, :])  # First get square part
    
    # Now compute the remaining rows of L
    if m > n:
        L_bottom = torch.zeros((m-n, n), dtype=A.dtype)
        A_bottom = P @ A[n:, :]  # Apply permutation to bottom rows
        
        for i in range(m-n):
            for j in range(n):
                if j == n-1:
                    # Last column just uses the value from A_bottom
                    L_bottom[i, j] = A_bottom[i, j]
                else:
                    # Earlier columns subtract the effect of previous columns
                    L_bottom[i, j] = A_bottom[i, j]
                    for k in range(j+1, n):
                        L_bottom[i, j] -= L_bottom[i, k] * U[j, k]
        
        # Combine top and bottom parts of L
        L = torch.cat([L_full, L_bottom], dim=0)
    else:
        L = L_full
    
    return L, U

def demonstrate_rectangular_lu():
    """Demonstrate LU decomposition for rectangular matrices."""
    # Create a rectangular matrix (more rows than columns)
    A = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])
    
    # Get dimensions
    m, n = A.shape
    
    # Compute LU decomposition
    L, U = lu_rectangular(A)
    
    # Display the matrices
    plot_matrix(A, f"Original Rectangular Matrix A ({m}x{n})")
    plot_matrix(L, f"Lower Triangular Matrix L ({m}x{n})")
    plot_matrix(U, f"Upper Triangular Matrix U ({n}x{n})")
    
    # Verify the decomposition
    LU = L @ U
    plot_matrix(LU, "Reconstructed Matrix LU")
    
    # Calculate reconstruction error
    error = torch.norm(A - LU).item()
    print(f"Reconstruction error: {error:.2e}")
    
    return A, L, U

# Demonstrate LU decomposition for rectangular matrices
A_rect, L_rect, U_rect = demonstrate_rectangular_lu()

# %% [markdown]
# ## 3. Optimized Implementations
# 
# In practice, LU decomposition is often implemented using optimized techniques for better performance and numerical stability. Let's explore some of these optimizations.
# 
# ### 3.1 Block LU Decomposition
# 
# Block LU decomposition breaks down a large matrix into smaller blocks, which can be processed more efficiently. For a matrix partitioned as:
# 
# $$A = \begin{pmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{pmatrix}$$
# 
# The block LU decomposition is:
# 
# $$A = \begin{pmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{pmatrix} \begin{pmatrix} U_{11} & U_{12} \\ 0 & U_{22} \end{pmatrix}$$
# 
# Let's implement a simple version of block LU decomposition:

# %%
def block_lu_decomposition(A, block_size=2):
    """
    Perform block LU decomposition.
    
    Args:
        A: Input square matrix
        block_size: Size of the blocks
        
    Returns:
        L: Lower triangular matrix
        U: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    assert n % block_size == 0, "Matrix size must be divisible by block_size"
    
    # Initialize L and U
    L = torch.eye(n, dtype=A.dtype)
    U = torch.zeros_like(A)
    
    # Process the matrix in blocks
    for i in range(0, n, block_size):
        # Compute U blocks in current block-row
        for j in range(0, i, block_size):
            U[i:i+block_size, j:j+block_size] = (
                A[i:i+block_size, j:j+block_size] - 
                L[i:i+block_size, :j] @ U[:j, j:j+block_size]
            )
        
        # Compute diagonal U block
        U[i:i+block_size, i:i+block_size] = (
            A[i:i+block_size, i:i+block_size] - 
            L[i:i+block_size, :i] @ U[:i, i:i+block_size]
        )
        
        # Compute L blocks in current block-column
        for j in range(i+block_size, n, block_size):
            L[j:j+block_size, i:i+block_size] = (
                A[j:j+block_size, i:i+block_size] - 
                L[j:j+block_size, :i] @ U[:i, i:i+block_size]
            ) @ torch.linalg.inv(U[i:i+block_size, i:i+block_size])
        
        # Compute U blocks in remaining rows of current block-column
        for j in range(i+block_size, n, block_size):
            U[i:i+block_size, j:j+block_size] = (
                A[i:i+block_size, j:j+block_size] - 
                L[i:i+block_size, :i] @ U[:i, j:j+block_size]
            )
    
    return L, U

def demonstrate_block_lu():
    """Demonstrate block LU decomposition."""
    # Create a 4x4 matrix for 2x2 blocks
    A = torch.tensor([
        [4.0, 1.0, 2.0, 3.0],
        [1.0, 5.0, 6.0, 2.0],
        [2.0, 6.0, 8.0, 4.0],
        [3.0, 2.0, 4.0, 7.0]
    ])
    
    # Compute block LU decomposition
    L_block, U_block = block_lu_decomposition(A, block_size=2)
    
    # For comparison, compute regular LU decomposition
    _, L_reg, U_reg = scipy.linalg.lu(A.numpy())
    
    # Display the matrices
    plot_matrix(A, "Original Matrix A")
    plot_matrix(L_block, "Block L")
    plot_matrix(U_block, "Block U")
    
    # Verify the decomposition
    LU_block = L_block @ U_block
    plot_matrix(LU_block, "Reconstructed Matrix (Block LU)")
    
    # Calculate reconstruction error
    error_block = torch.norm(A - LU_block).item()
    print(f"Block LU reconstruction error: {error_block:.2e}")
    
    # Compare with regular LU
    L_reg = torch.tensor(L_reg, dtype=torch.float64)
    U_reg = torch.tensor(U_reg, dtype=torch.float64)
    LU_reg = L_reg @ U_reg
    error_reg = torch.norm(A - LU_reg).item()
    print(f"Regular LU reconstruction error: {error_reg:.2e}")
    
    # Compare performance for larger matrices
    def time_comparison(n=100, block_size=25):
        # Create a larger random matrix
        A_large = torch.rand(n, n, dtype=torch.float64)
        
        # Time block LU
        start_time = time.time()
        L_block, U_block = block_lu_decomposition(A_large, block_size=block_size)
        block_time = time.time() - start_time
        
        # Time regular LU
        start_time = time.time()
        _, L_reg, U_reg = scipy.linalg.lu(A_large.numpy())
        reg_time = time.time() - start_time
        
        return block_time, reg_time
    
    # Compare timing
    block_time, reg_time = time_comparison()
    print(f"\nTiming for 100x100 matrix:")
    print(f"Block LU time: {block_time:.4f}s")
    print(f"Regular LU time: {reg_time:.4f}s")
    
    return A, L_block, U_block

# Demonstrate block LU decomposition
A_block, L_block, U_block = demonstrate_block_lu()

# %% [markdown]
# ### 3.2 Memory Layout Optimizations
# 
# Modern implementations of LU decomposition take advantage of optimized memory layouts for better cache performance. These implementations typically use BLAS (Basic Linear Algebra Subprograms) and LAPACK libraries, which are heavily optimized.
# 
# Let's discuss the importance of memory layout:

# %%
def demonstrate_memory_layout():
    """Demonstrate the importance of memory layout for LU decomposition."""
    n = 1000
    
    # Create random matrices in both row-major and column-major layout
    A_row_major = np.random.rand(n, n)  # NumPy uses row-major (C-style)
    A_col_major = np.asfortranarray(A_row_major)  # Convert to column-major (Fortran-style)
    
    # Time LU decomposition for row-major layout
    start_time = time.time()
    LU_row_major = scipy.linalg.lu_factor(A_row_major)
    row_major_time = time.time() - start_time
    
    # Time LU decomposition for column-major layout
    start_time = time.time()
    LU_col_major = scipy.linalg.lu_factor(A_col_major)
    col_major_time = time.time() - start_time
    
    # Compare timing
    print("Memory Layout Optimization:")
    print(f"Row-major (C-style) time: {row_major_time:.4f}s")
    print(f"Column-major (Fortran-style) time: {col_major_time:.4f}s")
    print(f"Speedup: {row_major_time / col_major_time:.2f}x")
    
    # Plot the timing comparison
    plt.figure(figsize=(8, 6))
    plt.bar(["Row-major", "Column-major"], [row_major_time, col_major_time])
    plt.ylabel("Time (seconds)")
    plt.title("LU Decomposition Time by Memory Layout")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Note on why column-major is faster for LU
    print("\nWhy is column-major layout often faster for LU decomposition?")
    print("LU decomposition processes the matrix column by column.")
    print("In column-major format, elements in a column are contiguous in memory,")
    print("leading to better cache locality and fewer cache misses.")
    
    return row_major_time, col_major_time

# Skip the actual execution for larger matrices to avoid long computation
# But show the conceptual explanation
print("Memory Layout Optimization:")
print("LU decomposition with column-major layout is typically faster")
print("because the algorithm processes columns sequentially.")
print("When elements in a column are contiguous in memory (column-major),")
print("this results in better cache performance.")

# %% [markdown]
# ## 4. Stability and Accuracy Considerations
# 
# Numerical stability is a critical aspect of LU decomposition. Let's explore some factors that affect the stability and accuracy of the algorithm.
# 
# ### 4.1 Condition Number Analysis
# 
# The condition number of a matrix affects the numerical stability of LU decomposition and the accuracy of solutions to linear systems.

# %%
def analyze_condition_number():
    """Analyze the effect of condition number on LU decomposition."""
    n = 10
    
    # Create matrices with different condition numbers
    condition_numbers = [1e1, 1e3, 1e5, 1e7, 1e9]
    errors = []
    residuals = []
    
    for kappa in condition_numbers:
        # Create a diagonal matrix with specified condition number
        s1 = 1.0
        sn = s1 / kappa
        S = np.diag(np.linspace(s1, sn, n))
        
        # Create a random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Create a matrix with the given condition number
        A = Q @ S @ Q.T
        
        # Create a right-hand side
        b = np.random.rand(n)
        
        # Compute LU decomposition
        try:
            lu, piv = scipy.linalg.lu_factor(A)
            x_lu = scipy.linalg.lu_solve((lu, piv), b)
            
            # Compute residual ||Ax - b||/||b||
            residual = np.linalg.norm(A @ x_lu - b) / np.linalg.norm(b)
            residuals.append(residual)
            
            # Compute error ||x - x_true||/||x_true|| if we know the true solution
            # (here we compute x_true using a more accurate method)
            x_true = np.linalg.solve(A, b)
            error = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)
            errors.append(error)
        except np.linalg.LinAlgError:
            errors.append(np.nan)
            residuals.append(np.nan)
    
    # Plot the results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog(condition_numbers, errors, 'o-', label='Solution Error')
    plt.loglog(condition_numbers, [1e-16 * k for k in condition_numbers], '--', label='Machine Epsilon × Condition Number')
    plt.xlabel("Condition Number")
    plt.ylabel("Relative Error")
    plt.title("Solution Error vs. Condition Number")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.loglog(condition_numbers, residuals, 'o-')
    plt.xlabel("Condition Number")
    plt.ylabel("Relative Residual")
    plt.title("Residual vs. Condition Number")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("Condition Number Analysis:")
    print("-" * 60)
    print(f"{'Condition Number':<20} {'Solution Error':<20} {'Residual':<20}")
    print("-" * 60)
    
    for kappa, error, residual in zip(condition_numbers, errors, residuals):
        print(f"{kappa:<20.1e} {error:<20.2e} {residual:<20.2e}")
    
    print("\nObservation:")
    print("The solution error grows approximately linearly with the condition number,")
    print("following the theoretical bound: error ≈ machine_epsilon × condition_number.")

# Demonstrate condition number effect
analyze_condition_number()

# %% [markdown]
# ### 4.2 Comparison of Pivoting Strategies
# 
# Different pivoting strategies affect the stability of LU decomposition. Let's compare partial pivoting (which we've used) with complete pivoting:

# %%
def compare_pivoting_strategies():
    """Compare different pivoting strategies for LU decomposition."""
    # Create a challenging matrix
    n = 5
    A = torch.tensor([
        [1e-10, 1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 2.0, 3.0, 4.0],
        [2.0, 2.0, 1.0, 3.0, 4.0],
        [3.0, 3.0, 3.0, 1.0, 4.0],
        [4.0, 4.0, 4.0, 4.0, 1.0]
    ])
    
    # No pivoting (will likely be unstable)
    def lu_no_pivoting(A):
        A_np = A.numpy()
        n = A_np.shape[0]
        L = np.eye(n)
        U = A_np.copy()
        
        for k in range(n-1):
            for i in range(k+1, n):
                if abs(U[k, k]) < 1e-10:
                    return None, None  # Pivot too small
                
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]
        
        return L, U
    
    # Partial pivoting (row exchanges)
    P, L_partial, U_partial = scipy.linalg.lu(A.numpy())
    
    # Calculate growth factor for partial pivoting
    growth_partial = np.max(np.abs(L_partial @ U_partial)) / np.max(np.abs(A.numpy()))
    
    # Try no pivoting
    try:
        L_no, U_no = lu_no_pivoting(A)
        if L_no is None:
            print("LU without pivoting failed due to a near-zero pivot")
            growth_no = float('inf')
        else:
            growth_no = np.max(np.abs(L_no @ U_no)) / np.max(np.abs(A.numpy()))
    except Exception as e:
        print(f"LU without pivoting failed: {e}")
        growth_no = float('inf')
        L_no, U_no = np.eye(n), np.zeros((n, n))
    
    # Display the matrices
    plot_matrix(A, "Original Matrix A")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    try:
        sns.heatmap(L_no, annot=True, fmt=".2e", cmap=blue_cmap)
        plt.title("L (No Pivoting)")
    except:
        plt.title("L (No Pivoting) - Failed")
    
    plt.subplot(1, 3, 2)
    try:
        sns.heatmap(U_no, annot=True, fmt=".2e", cmap=blue_cmap)
        plt.title("U (No Pivoting)")
    except:
        plt.title("U (No Pivoting) - Failed")
    
    plt.subplot(1, 3, 3)
    try:
        sns.heatmap(L_no @ U_no, annot=True, fmt=".2e", cmap=blue_cmap)
        plt.title(f"LU (No Pivoting)\nGrowth Factor: {growth_no:.2e}")
    except:
        plt.title("LU (No Pivoting) - Failed")
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(L_partial, annot=True, fmt=".2e", cmap=blue_cmap)
    plt.title("L (Partial Pivoting)")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(U_partial, annot=True, fmt=".2e", cmap=blue_cmap)
    plt.title("U (Partial Pivoting)")
    
    plt.subplot(1, 3, 3)
    sns.heatmap(L_partial @ U_partial, annot=True, fmt=".2e", cmap=blue_cmap)
    plt.title(f"LU (Partial Pivoting)\nGrowth Factor: {growth_partial:.2e}")
    
    plt.tight_layout()
    plt.show()
    
    # Compare error in solving linear systems
    b = torch.ones(n, dtype=torch.float64)
    
    try:
        # Solve with no pivoting (if it succeeded)
        if L_no is not None:
            y_no = np.linalg.solve(L_no, b.numpy())
            x_no = np.linalg.solve(U_no, y_no)
            residual_no = np.linalg.norm(A.numpy() @ x_no - b.numpy()) / np.linalg.norm(b.numpy())
        else:
            residual_no = float('inf')
    except:
        residual_no = float('inf')
    
    # Solve with partial pivoting
    lu_partial, piv_partial = scipy.linalg.lu_factor(A.numpy())
    x_partial = scipy.linalg.lu_solve((lu_partial, piv_partial), b.numpy())
    residual_partial = np.linalg.norm(A.numpy() @ x_partial - b.numpy()) / np.linalg.norm(b.numpy())
    
    # Print comparison
    print("Pivoting Strategy Comparison:")
    print("-" * 60)
    print(f"{'Strategy':<20} {'Growth Factor':<20} {'Residual':<20}")
    print("-" * 60)
    print(f"{'No Pivoting':<20} {growth_no:<20.2e} {residual_no:<20.2e}")
    print(f"{'Partial Pivoting':<20} {growth_partial:<20.2e} {residual_partial:<20.2e}")
    
    print("\nObservation:")
    print("Partial pivoting significantly improves numerical stability,")
    print("especially for matrices with small pivots in the diagonal.")
    
    return A, growth_no, growth_partial, residual_no, residual_partial

# Compare pivoting strategies
A_piv, growth_no, growth_partial, res_no, res_partial = compare_pivoting_strategies()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored the mathematical theory and practical implementations of LU decomposition:
# 
# 1. We examined the conditions for the existence of LU factorization without pivoting.
# 
# 2. We implemented and demonstrated various variants of LU decomposition:
#    - LDU decomposition (separating the diagonal elements)
#    - PLU decomposition (with partial pivoting)
#    - LU factorization for rectangular matrices
# 
# 3. We discussed optimization techniques for LU decomposition:
#    - Block LU decomposition for better performance
#    - Memory layout considerations (column-major vs. row-major)
# 
# 4. We analyzed numerical stability aspects:
#    - The effect of condition number on solution accuracy
#    - Comparison of different pivoting strategies
# 
# These insights provide a deeper understanding of LU decomposition beyond the basic algorithm, helping to choose the appropriate variant and implementation for specific applications.
# 
# In the next notebook, we'll explore practical applications of LU decomposition in various domains.

# %%