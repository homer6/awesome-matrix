# %% [markdown]
# # Cholesky Decomposition: Applications
# 
# In this notebook, we explore practical applications of Cholesky decomposition, including:
# 
# 1. Solving linear systems
# 2. Least squares problems
# 3. Matrix inversion
# 4. Sampling from multivariate Gaussian distributions
# 5. Optimization algorithms
# 6. Kalman filtering
# 
# These applications demonstrate the versatility and efficiency of Cholesky decomposition in various computational tasks.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set the default style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Helper function to create positive definite matrices
def create_positive_definite_matrix(n, method="random", condition_number=None):
    """Create a positive definite matrix of size n×n."""
    if method == "random":
        # Create a random matrix
        B = torch.randn(n, n)
        # A = B*B^T is guaranteed to be positive definite (if B is full rank)
        A = B @ B.T
        # Add a small value to the diagonal to ensure positive definiteness
        A = A + torch.eye(n) * 1e-5
        
        # If a specific condition number is requested
        if condition_number is not None:
            # Get eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(A)
            # Adjust eigenvalues to get the desired condition number
            min_eig = 1.0
            max_eig = condition_number
            eigenvalues = torch.linspace(min_eig, max_eig, n)
            # Reconstruct the matrix
            A = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
            
        return A
    elif method == "predetermined":
        # A predefined example
        A = torch.tensor([[4.0, 1.0, 1.0], 
                          [1.0, 3.0, 2.0], 
                          [1.0, 2.0, 6.0]])
        return A
    else:
        raise ValueError("Unknown method")

# Helper function to visualize matrices
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

# %% [markdown]
# ## 1. Solving Linear Systems
# 
# One of the most common applications of Cholesky decomposition is solving linear systems of the form $Ax = b$, where $A$ is a symmetric positive definite matrix.
# 
# The solution involves two triangular solves:
# 1. Decompose $A = LL^T$ using Cholesky
# 2. Solve $Ly = b$ for $y$ using forward substitution
# 3. Solve $L^Tx = y$ for $x$ using backward substitution
# 
# This approach is numerically stable and efficient, requiring only $O(n^2)$ operations for the triangular solves.

# %%
def solve_linear_system_cholesky(A, b):
    """
    Solve a linear system Ax = b using Cholesky decomposition.
    
    Parameters:
        A (torch.Tensor): Symmetric positive definite matrix
        b (torch.Tensor): Right-hand side vector
        
    Returns:
        x (torch.Tensor): Solution vector
    """
    # Step 1: Compute Cholesky decomposition A = L*L^T
    L = torch.linalg.cholesky(A)
    
    # Step 2: Solve Ly = b using forward substitution
    y = torch.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - torch.sum(L[i, :i] * y[:i])) / L[i, i]
    
    # Step 3: Solve L^T x = y using backward substitution
    x = torch.zeros_like(y)
    for i in range(len(y)-1, -1, -1):
        x[i] = (y[i] - torch.sum(L[i+1:, i] * x[i+1:])) / L[i, i]
    
    return x

# Create a system to solve
n = 4
A = create_positive_definite_matrix(n, method="random")
x_true = torch.randn(n)  # True solution
b = A @ x_true  # Right-hand side

# Solve using our Cholesky implementation
x_cholesky = solve_linear_system_cholesky(A, b)

# Solve using PyTorch's linear solver for comparison
x_torch = torch.linalg.solve(A, b)

print("True solution:", x_true)
print("\nCholesky solution:", x_cholesky)
print("PyTorch direct solution:", x_torch)
print("\nError in Cholesky solution:", torch.norm(x_cholesky - x_true).item())
print("Error in PyTorch solution:", torch.norm(x_torch - x_true).item())

# %% [markdown]
# Let's visualize the solution process:

# %%
def visualize_linear_system(A, b, L, y, x):
    """Visualize the steps in solving a linear system using Cholesky."""
    plt.figure(figsize=(15, 6))
    
    # Original system
    plt.subplot(1, 3, 1)
    system_matrix = torch.zeros(n, n+1)
    system_matrix[:, :n] = A
    system_matrix[:, n] = b
    sns.heatmap(system_matrix.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5,
               xticklabels=list(range(n)) + ['b'])
    plt.title("Original System: Ax = b")
    
    # Step 1: Forward substitution
    plt.subplot(1, 3, 2)
    forward_matrix = torch.zeros(n, n+1)
    forward_matrix[:, :n] = L
    forward_matrix[:, n] = y
    mask = torch.triu(torch.ones(n, n), diagonal=1)
    mask = torch.cat([mask, torch.zeros(n, 1)], dim=1)
    sns.heatmap(forward_matrix.numpy(), annot=True, fmt=".2f", cmap="Greens", linewidths=.5,
               mask=mask.bool().numpy(), xticklabels=list(range(n)) + ['y'])
    plt.title("Step 1: Solve Ly = b")
    
    # Step 2: Backward substitution
    plt.subplot(1, 3, 3)
    backward_matrix = torch.zeros(n, n+1)
    backward_matrix[:, :n] = L.T
    backward_matrix[:, n] = x
    mask = torch.tril(torch.ones(n, n), diagonal=-1)
    mask = torch.cat([mask, torch.zeros(n, 1)], dim=1)
    sns.heatmap(backward_matrix.numpy(), annot=True, fmt=".2f", cmap="Oranges", linewidths=.5,
               mask=mask.bool().numpy(), xticklabels=list(range(n)) + ['x'])
    plt.title("Step 2: Solve L^T x = y")
    
    plt.tight_layout()
    plt.show()

# Compute intermediate results for visualization
L = torch.linalg.cholesky(A)
y = torch.triangular_solve(b.unsqueeze(1), L, upper=False)[0].squeeze()

# Visualize the solution process
visualize_linear_system(A, b, L, y, x_cholesky)

# %% [markdown]
# ### Comparing Efficiency with Different Methods
# 
# Let's compare the efficiency of solving linear systems using Cholesky decomposition versus other methods:

# %%
def benchmark_linear_solvers(sizes):
    """Benchmark different linear solvers for systems of various sizes."""
    cholesky_times = []
    lu_times = []
    direct_times = []
    
    for n in sizes:
        # Create a well-conditioned system
        A = create_positive_definite_matrix(n, condition_number=10)
        b = torch.randn(n)
        
        # Method 1: Cholesky decomposition
        start_time = time.time()
        L = torch.linalg.cholesky(A)
        y = torch.triangular_solve(b.unsqueeze(1), L, upper=False)[0].squeeze()
        x_cholesky = torch.triangular_solve(y.unsqueeze(1), L.T, upper=True)[0].squeeze()
        cholesky_time = time.time() - start_time
        cholesky_times.append(cholesky_time)
        
        # Method 2: LU decomposition
        start_time = time.time()
        LU, pivots = torch.linalg.lu_factor(A)
        x_lu = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(1)).squeeze()
        lu_time = time.time() - start_time
        lu_times.append(lu_time)
        
        # Method 3: Direct solve
        start_time = time.time()
        x_direct = torch.linalg.solve(A, b)
        direct_time = time.time() - start_time
        direct_times.append(direct_time)
        
        print(f"Size {n}×{n}:")
        print(f"  Cholesky: {cholesky_time:.6f} seconds")
        print(f"  LU: {lu_time:.6f} seconds")
        print(f"  Direct: {direct_time:.6f} seconds")
    
    return cholesky_times, lu_times, direct_times

# Benchmark with different matrix sizes
sizes = [10, 50, 100, 500, 1000]
cholesky_times, lu_times, direct_times = benchmark_linear_solvers(sizes)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sizes, cholesky_times, 'o-', label='Cholesky')
plt.plot(sizes, lu_times, 's-', label='LU')
plt.plot(sizes, direct_times, 'x-', label='Direct')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Computation Time (seconds)')
plt.title('Linear System Solver Performance')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### Effect of Condition Number on Solver Stability
# 
# The condition number of a matrix affects the numerical stability of linear system solvers. 
# Let's examine how different methods perform with varying condition numbers:

# %%
def compare_solvers_stability(condition_numbers):
    """Compare solver stability for matrices with different condition numbers."""
    cholesky_errors = []
    lu_errors = []
    direct_errors = []
    
    n = 100  # Fixed size
    
    for cond in condition_numbers:
        # Create a matrix with specific condition number
        A = create_positive_definite_matrix(n, condition_number=cond)
        x_true = torch.randn(n)
        b = A @ x_true
        
        # Method 1: Cholesky decomposition
        try:
            L = torch.linalg.cholesky(A)
            y = torch.triangular_solve(b.unsqueeze(1), L, upper=False)[0].squeeze()
            x_cholesky = torch.triangular_solve(y.unsqueeze(1), L.T, upper=True)[0].squeeze()
            error_cholesky = torch.norm(x_cholesky - x_true) / torch.norm(x_true)
        except:
            error_cholesky = float('nan')
        cholesky_errors.append(error_cholesky)
        
        # Method 2: LU decomposition
        try:
            LU, pivots = torch.linalg.lu_factor(A)
            x_lu = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(1)).squeeze()
            error_lu = torch.norm(x_lu - x_true) / torch.norm(x_true)
        except:
            error_lu = float('nan')
        lu_errors.append(error_lu)
        
        # Method 3: Direct solve
        try:
            x_direct = torch.linalg.solve(A, b)
            error_direct = torch.norm(x_direct - x_true) / torch.norm(x_true)
        except:
            error_direct = float('nan')
        direct_errors.append(error_direct)
        
        print(f"Condition Number: {cond}")
        print(f"  Cholesky relative error: {error_cholesky:.6e}")
        print(f"  LU relative error: {error_lu:.6e}")
        print(f"  Direct relative error: {error_direct:.6e}")
    
    return cholesky_errors, lu_errors, direct_errors

# Compare with different condition numbers
condition_numbers = [1, 10, 100, 1000, 10000, 100000]
cholesky_errors, lu_errors, direct_errors = compare_solvers_stability(condition_numbers)

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogy(condition_numbers, cholesky_errors, 'o-', label='Cholesky')
plt.semilogy(condition_numbers, lu_errors, 's-', label='LU')
plt.semilogy(condition_numbers, direct_errors, 'x-', label='Direct')
plt.xlabel('Condition Number')
plt.ylabel('Relative Error (log scale)')
plt.title('Solver Stability vs. Condition Number')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 2. Least Squares Problems
# 
# Cholesky decomposition is efficient for solving least squares problems, particularly when the normal equations approach is used.
# 
# Given an overdetermined system $Ax = b$ where $A$ is an $m \times n$ matrix with $m > n$, the least squares solution minimizes $\|Ax - b\|_2^2$. 
# This is solved by the normal equations: $A^TAx = A^Tb$.
# 
# Since $A^TA$ is symmetric positive definite (if $A$ has full column rank), we can use Cholesky decomposition to solve this efficiently.

# %%
def solve_least_squares_cholesky(A, b):
    """
    Solve a least squares problem using normal equations and Cholesky decomposition.
    
    Parameters:
        A (torch.Tensor): Coefficient matrix (m x n with m > n)
        b (torch.Tensor): Right-hand side vector (m)
        
    Returns:
        x (torch.Tensor): Least squares solution (n)
    """
    # Form normal equations: A^T A x = A^T b
    ATA = A.T @ A
    ATb = A.T @ b
    
    # Solve using Cholesky decomposition
    L = torch.linalg.cholesky(ATA)
    
    # Forward substitution
    y = torch.zeros_like(ATb)
    for i in range(len(y)):
        y[i] = (ATb[i] - torch.sum(L[i, :i] * y[:i])) / L[i, i]
    
    # Backward substitution
    x = torch.zeros_like(y)
    for i in range(len(x)-1, -1, -1):
        x[i] = (y[i] - torch.sum(L[i+1:, i] * x[i+1:])) / L[i, i]
    
    return x

# Create an overdetermined system
m, n = 10, 4  # More equations than unknowns
A = torch.randn(m, n)
x_true = torch.randn(n)
b = A @ x_true + 0.1 * torch.randn(m)  # Add some noise

# Solve using Cholesky
x_cholesky = solve_least_squares_cholesky(A, b)

# Solve using PyTorch's least squares solver for comparison
x_torch, _ = torch.linalg.lstsq(A, b.unsqueeze(1))
x_torch = x_torch.squeeze()

print("True solution:", x_true)
print("\nCholesky least squares solution:", x_cholesky)
print("PyTorch least squares solution:", x_torch)
print("\nError in Cholesky solution:", torch.norm(x_cholesky - x_true).item())
print("Error in PyTorch solution:", torch.norm(x_torch - x_true).item())

# %% [markdown]
# Let's visualize the least squares solution:

# %%
def visualize_least_squares(A, b, x_sol):
    """Visualize a least squares problem and its solution."""
    # Only work with 2D cases for visualization
    if A.shape[1] != 2:
        return
    
    plt.figure(figsize=(10, 8))
    
    # Plot the data points
    plt.scatter(A[:, 0].numpy(), b.numpy(), color='blue', label='Data Points')
    
    # Plot the fitted line/plane
    x_range = torch.linspace(A[:, 0].min(), A[:, 0].max(), 100)
    if A.shape[1] == 2:  # If we have two parameters
        y_fit = x_sol[0] * x_range + x_sol[1] * A[:, 1].mean()
        plt.plot(x_range.numpy(), y_fit.numpy(), 'r-', linewidth=2, label='Least Squares Fit')
    
    # Plot the residuals
    residuals = b - A @ x_sol
    for i in range(len(b)):
        plt.plot([A[i, 0], A[i, 0]], [b[i], b[i] - residuals[i]], 'g--', alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Fitting')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the residuals
    plt.figure(figsize=(10, 4))
    plt.stem(range(len(residuals)), residuals.numpy())
    plt.xlabel('Data Point')
    plt.ylabel('Residual')
    plt.title('Residuals')
    plt.grid(True)
    plt.show()

# Create a 2D problem for visualization
m = 20
A_2d = torch.zeros(m, 2)
A_2d[:, 0] = torch.linspace(-5, 5, m)
A_2d[:, 1] = 1.0  # Intercept term
x_true_2d = torch.tensor([2.5, 1.5])  # Slope and intercept
b_2d = A_2d @ x_true_2d + 0.5 * torch.randn(m)  # Add some noise

# Solve using Cholesky
x_cholesky_2d = solve_least_squares_cholesky(A_2d, b_2d)

# Visualize
visualize_least_squares(A_2d, b_2d, x_cholesky_2d)

# %% [markdown]
# ### Comparing Normal Equations vs QR for Least Squares
# 
# While Cholesky decomposition provides an efficient way to solve least squares problems via normal equations, the QR decomposition approach is often more numerically stable, especially for ill-conditioned problems. Let's compare both approaches:

# %%
def compare_least_squares_methods(condition_numbers):
    """Compare normal equations (Cholesky) vs QR for least squares with different condition numbers."""
    cholesky_errors = []
    qr_errors = []
    
    m, n = 100, 10  # Fix the problem size
    
    for cond in condition_numbers:
        # Create a matrix with specific condition structure
        U, _, V = torch.svd(torch.randn(m, n))
        singular_values = torch.logspace(0, -np.log10(cond), n)
        A = U[:, :n] @ torch.diag(singular_values) @ V.T
        
        x_true = torch.randn(n)
        b = A @ x_true + 0.01 * torch.randn(m)  # Add some noise
        
        # Method 1: Normal equations with Cholesky
        try:
            ATA = A.T @ A
            ATb = A.T @ b
            L = torch.linalg.cholesky(ATA)
            y = torch.triangular_solve(ATb.unsqueeze(1), L, upper=False)[0].squeeze()
            x_cholesky = torch.triangular_solve(y.unsqueeze(1), L.T, upper=True)[0].squeeze()
            error_cholesky = torch.norm(x_cholesky - x_true) / torch.norm(x_true)
        except:
            error_cholesky = float('nan')
        cholesky_errors.append(error_cholesky)
        
        # Method 2: QR decomposition
        try:
            Q, R = torch.linalg.qr(A)
            x_qr = torch.triangular_solve((Q.T @ b).unsqueeze(1), R, upper=True)[0].squeeze()
            error_qr = torch.norm(x_qr - x_true) / torch.norm(x_true)
        except:
            error_qr = float('nan')
        qr_errors.append(error_qr)
        
        print(f"Condition Number: {cond}")
        print(f"  Cholesky relative error: {error_cholesky:.6e}")
        print(f"  QR relative error: {error_qr:.6e}")
    
    return cholesky_errors, qr_errors

# Compare with different condition numbers
condition_numbers = [1, 10, 100, 1000, 10000, 100000]
cholesky_errors, qr_errors = compare_least_squares_methods(condition_numbers)

# Plot the results
plt.figure(figsize=(10, 6))
plt.semilogy(condition_numbers, cholesky_errors, 'o-', label='Normal Equations (Cholesky)')
plt.semilogy(condition_numbers, qr_errors, 's-', label='QR Decomposition')
plt.xlabel('Condition Number')
plt.ylabel('Relative Error (log scale)')
plt.title('Least Squares Methods: Stability vs. Condition Number')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## 3. Matrix Inversion
# 
# Cholesky decomposition can be used for efficient matrix inversion when the matrix is symmetric positive definite. This involves:
# 1. Computing the Cholesky decomposition $A = LL^T$
# 2. Inverting the triangular factor $L$ to get $L^{-1}$
# 3. Computing $A^{-1} = (L^{-1})^T L^{-1}$

# %%
def invert_matrix_cholesky(A):
    """
    Compute the inverse of a symmetric positive definite matrix using Cholesky decomposition.
    
    Parameters:
        A (torch.Tensor): Symmetric positive definite matrix
        
    Returns:
        A_inv (torch.Tensor): Inverse of A
    """
    n = A.shape[0]
    
    # Step 1: Compute Cholesky decomposition A = L*L^T
    L = torch.linalg.cholesky(A)
    
    # Step 2: Compute inverse of L (lower triangular)
    L_inv = torch.zeros_like(L)
    for i in range(n):
        L_inv[i, i] = 1.0 / L[i, i]
        for j in range(i-1, -1, -1):
            L_inv[i, j] = -torch.sum(L_inv[i, j+1:i+1] * L[j+1:i+1, j]) / L[j, j]
    
    # Step 3: Compute A^{-1} = (L^{-1})^T * L^{-1}
    A_inv = L_inv.T @ L_inv
    
    return A_inv

# Create a symmetric positive definite matrix
n = 4
A = create_positive_definite_matrix(n)

# Invert using Cholesky
A_inv_cholesky = invert_matrix_cholesky(A)

# Invert using PyTorch's inverse function for comparison
A_inv_torch = torch.inverse(A)

print("Matrix A:")
print(A)
print("\nInverse using Cholesky:")
print(A_inv_cholesky)
print("\nInverse using PyTorch:")
print(A_inv_torch)

# Verify: A * A^{-1} should be close to the identity matrix
I_cholesky = A @ A_inv_cholesky
I_torch = A @ A_inv_torch

print("\nA * A^{-1} (Cholesky):")
print(I_cholesky)
print("\nA * A^{-1} (PyTorch):")
print(I_torch)

print("\nError in Cholesky inversion:", torch.norm(I_cholesky - torch.eye(n)).item())
print("Error in PyTorch inversion:", torch.norm(I_torch - torch.eye(n)).item())

# %% [markdown]
# Let's visualize the steps in matrix inversion:

# %%
def visualize_matrix_inversion(A, L, L_inv, A_inv):
    """Visualize the steps in matrix inversion using Cholesky."""
    plt.figure(figsize=(15, 5))
    
    # Original matrix
    plt.subplot(1, 4, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title("Original Matrix A")
    
    # Cholesky factor
    plt.subplot(1, 4, 2)
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap="Greens", linewidths=.5)
    plt.title("Cholesky Factor L")
    
    # Inverse of Cholesky factor
    plt.subplot(1, 4, 3)
    sns.heatmap(L_inv.numpy(), annot=True, fmt=".2f", cmap="Oranges", linewidths=.5)
    plt.title("Inverse of L")
    
    # Inverse of original matrix
    plt.subplot(1, 4, 4)
    sns.heatmap(A_inv.numpy(), annot=True, fmt=".2f", cmap="Purples", linewidths=.5)
    plt.title("Inverse of A")
    
    plt.tight_layout()
    plt.show()

# Compute intermediate results for visualization
L = torch.linalg.cholesky(A)
L_inv = torch.triangular_solve(torch.eye(n), L, upper=False)[0]

# Visualize the inversion process
visualize_matrix_inversion(A, L, L_inv, A_inv_cholesky)

# %% [markdown]
# ## 4. Sampling from Multivariate Gaussian Distributions
# 
# Cholesky decomposition is often used for generating samples from a multivariate Gaussian distribution.
# 
# To generate samples from $\mathcal{N}(\mu, \Sigma)$, we:
# 1. Compute the Cholesky decomposition $\Sigma = LL^T$
# 2. Generate a vector $z$ of independent standard normal samples
# 3. Compute $x = \mu + Lz$, which will have the desired distribution

# %%
def sample_multivariate_gaussian(mu, cov, num_samples=1000):
    """
    Generate samples from a multivariate Gaussian distribution.
    
    Parameters:
        mu (torch.Tensor): Mean vector
        cov (torch.Tensor): Covariance matrix
        num_samples (int): Number of samples to generate
        
    Returns:
        samples (torch.Tensor): Generated samples
    """
    # Get dimensions
    n = len(mu)
    
    # Compute Cholesky decomposition of covariance matrix
    L = torch.linalg.cholesky(cov)
    
    # Generate independent standard normal samples
    z = torch.randn(num_samples, n)
    
    # Transform to desired distribution: x = mu + L*z
    samples = mu + z @ L.T
    
    return samples

# Define a 2D Gaussian for visualization
mu = torch.tensor([1.0, 2.0])
cov = torch.tensor([[2.0, 0.5], 
                     [0.5, 1.0]])

# Generate samples
samples = sample_multivariate_gaussian(mu, cov, num_samples=1000)

# Visualize the samples and distribution
plt.figure(figsize=(10, 8))

# Plot the samples
plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), alpha=0.5)
plt.plot(mu[0].item(), mu[1].item(), 'ro', markersize=10, label='Mean')

# Plot 95% confidence ellipse
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    # Compute the covariance matrix
    cov = torch.cov(torch.stack([x, y]))
    pearson = cov[0, 1] / torch.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using the Cholesky decomposition, we can get the "standard deviation"
    ell_radius_x = torch.sqrt(1 + pearson) * torch.sqrt(cov[0, 0]) * n_std
    ell_radius_y = torch.sqrt(1 - pearson) * torch.sqrt(cov[1, 1]) * n_std
    
    ellipse = Ellipse((x.mean(), y.mean()), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      **kwargs)
    
    # Compute the axis angles
    if cov[0, 0] < cov[1, 1]:
        # Horizontal axis is the shorter one
        ell_theta = 0.5 * torch.arctan(2 * cov[0, 1] / (cov[1, 1] - cov[0, 0])).item()
    else:
        # Vertical axis is the shorter one
        ell_theta = 0.5 * torch.arctan(2 * cov[0, 1] / (cov[1, 1] - cov[0, 0])).item() + np.pi/2
    
    # Rotate the ellipse
    ellipse.angle = ell_theta * 180 / np.pi
    
    return ax.add_patch(ellipse)

# Plot the confidence ellipse
confidence_ellipse(samples[:, 0], samples[:, 1], plt.gca(), n_std=2.0,
                  edgecolor='red', facecolor='none', label='95% Confidence Region')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Samples from Multivariate Gaussian Distribution')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# ### Visualizing the Cholesky Approach
# 
# Let's visualize how the Cholesky decomposition transforms standard normal samples:

# %%
def visualize_gaussian_sampling(mu, cov, num_samples=500):
    """Visualize the process of sampling from a multivariate Gaussian."""
    n = len(mu)
    L = torch.linalg.cholesky(cov)
    
    # Step 1: Generate standard normal samples
    z = torch.randn(num_samples, n)
    
    # Step 2: Transform using Cholesky factor
    samples = mu + z @ L.T
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot standard normal samples
    plt.subplot(1, 3, 1)
    plt.scatter(z[:, 0].numpy(), z[:, 1].numpy(), alpha=0.5)
    plt.title('Step 1: Standard Normal Samples z')
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.axis('equal')
    plt.grid(True)
    
    # Plot the effect of the Cholesky factor
    plt.subplot(1, 3, 2)
    transformed = z @ L.T
    plt.scatter(transformed[:, 0].numpy(), transformed[:, 1].numpy(), alpha=0.5)
    plt.title('Step 2: After Applying Cholesky Factor L')
    plt.xlabel('(Lz)₁')
    plt.ylabel('(Lz)₂')
    plt.axis('equal')
    plt.grid(True)
    
    # Plot the final samples
    plt.subplot(1, 3, 3)
    plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), alpha=0.5)
    plt.plot(mu[0].item(), mu[1].item(), 'ro', markersize=10)
    plt.title('Step 3: Final Samples x = μ + Lz')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.axis('equal')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the Cholesky factor
    plt.figure(figsize=(8, 6))
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title('Cholesky Factor L of Covariance Matrix')
    plt.tight_layout()
    plt.show()

# Visualize the sampling process
visualize_gaussian_sampling(mu, cov)

# %% [markdown]
# ## 5. Optimization Algorithms
# 
# Cholesky decomposition is used in many optimization algorithms, particularly in Newton and quasi-Newton methods. Here, we'll implement a simple Newton's method for unconstrained optimization that uses Cholesky decomposition to compute the search direction.

# %%
def newton_method_cholesky(f, grad_f, hessian_f, x0, tol=1e-6, max_iter=50):
    """
    Newton's method for unconstrained optimization using Cholesky decomposition.
    
    Parameters:
        f (function): Objective function to minimize
        grad_f (function): Gradient of the objective function
        hessian_f (function): Hessian of the objective function
        x0 (torch.Tensor): Initial point
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        
    Returns:
        x (torch.Tensor): Optimal point
        f_values (list): Function values at each iteration
    """
    x = x0.clone()
    f_values = [f(x)]
    grad_norm_values = [torch.norm(grad_f(x)).item()]
    
    for i in range(max_iter):
        # Compute gradient and Hessian
        g = grad_f(x)
        H = hessian_f(x)
        
        # Check for convergence
        grad_norm = torch.norm(g)
        if grad_norm < tol:
            print(f"Converged after {i} iterations")
            break
        
        # Compute Newton direction: Solve H * p = -g
        try:
            L = torch.linalg.cholesky(H)
            y = torch.triangular_solve(-g.unsqueeze(1), L, upper=False)[0].squeeze()
            p = torch.triangular_solve(y.unsqueeze(1), L.T, upper=True)[0].squeeze()
        except:
            # If Hessian is not positive definite, add regularization
            H_reg = H + 1e-3 * torch.eye(len(x))
            L = torch.linalg.cholesky(H_reg)
            y = torch.triangular_solve(-g.unsqueeze(1), L, upper=False)[0].squeeze()
            p = torch.triangular_solve(y.unsqueeze(1), L.T, upper=True)[0].squeeze()
        
        # Line search (simple backtracking)
        alpha = 1.0
        while f(x + alpha * p) > f(x) + 1e-4 * alpha * g.dot(p) and alpha > 1e-10:
            alpha *= 0.5
        
        # Update
        x = x + alpha * p
        
        # Record function value
        f_values.append(f(x))
        grad_norm_values.append(grad_norm.item())
        
        print(f"Iteration {i+1}: f(x) = {f_values[-1]:.6f}, |grad f(x)| = {grad_norm_values[-1]:.6f}, alpha = {alpha:.6f}")
    
    return x, f_values, grad_norm_values

# Define a simple quadratic function for testing
def f(x):
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([-1.0, -2.0])
    return 0.5 * x.dot(A @ x) + b.dot(x)

def grad_f(x):
    A = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    b = torch.tensor([-1.0, -2.0])
    return A @ x + b

def hessian_f(x):
    return torch.tensor([[2.0, 0.5], [0.5, 1.0]])

# Starting point
x0 = torch.tensor([3.0, 2.0])

# Run Newton's method
x_opt, f_values, grad_norm_values = newton_method_cholesky(f, grad_f, hessian_f, x0)

print("\nOptimal point:", x_opt)
print("Optimal value:", f(x_opt))

# Visualize the optimization process
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.semilogy(range(len(f_values)), f_values, 'o-')
plt.xlabel('Iteration')
plt.ylabel('Function Value (log scale)')
plt.title('Convergence of Function Values')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.semilogy(range(len(grad_norm_values)), grad_norm_values, 'o-')
plt.xlabel('Iteration')
plt.ylabel('Gradient Norm (log scale)')
plt.title('Convergence of Gradient Norm')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Kalman Filtering
# 
# Cholesky decomposition is also used in Kalman filtering, particularly for the covariance update step. Here, we'll implement a simple Kalman filter for a 1D tracking problem.

# %%
def kalman_filter_cholesky(z, x0, P0, F, H, Q, R):
    """
    Kalman filter implementation using Cholesky decomposition for the covariance updates.
    
    Parameters:
        z (torch.Tensor): Measurements, shape (time_steps, measurement_dim)
        x0 (torch.Tensor): Initial state estimate
        P0 (torch.Tensor): Initial state covariance
        F (torch.Tensor): State transition matrix
        H (torch.Tensor): Measurement matrix
        Q (torch.Tensor): Process noise covariance
        R (torch.Tensor): Measurement noise covariance
        
    Returns:
        x_filtered (torch.Tensor): Filtered state estimates
        P_filtered (torch.Tensor): Filtered state covariances
    """
    time_steps = z.shape[0]
    state_dim = x0.shape[0]
    
    # Initialize arrays to store results
    x_filtered = torch.zeros(time_steps+1, state_dim)
    P_filtered = torch.zeros(time_steps+1, state_dim, state_dim)
    
    # Initial state
    x_filtered[0] = x0
    P_filtered[0] = P0
    
    for t in range(1, time_steps+1):
        # Prediction step
        x_pred = F @ x_filtered[t-1]
        P_pred = F @ P_filtered[t-1] @ F.T + Q
        
        # Update step with Cholesky decomposition
        S = H @ P_pred @ H.T + R  # Innovation covariance
        
        # Compute Kalman gain using Cholesky decomposition
        L_S = torch.linalg.cholesky(S)
        temp = torch.triangular_solve((H @ P_pred).T.unsqueeze(2), L_S, upper=False)[0].squeeze()
        K = torch.triangular_solve(temp.unsqueeze(2), L_S.T, upper=True)[0].squeeze().T
        
        # Update state and covariance
        innovation = z[t-1] - H @ x_pred
        x_filtered[t] = x_pred + K @ innovation
        
        # Joseph form for covariance update (more stable)
        I = torch.eye(state_dim)
        P_filtered[t] = (I - K @ H) @ P_pred @ (I - K @ H).T + K @ R @ K.T
        
    return x_filtered, P_filtered

# Example: 1D tracking with position and velocity
def generate_tracking_data(time_steps=100, dt=0.1, process_noise_std=0.1, measurement_noise_std=0.5):
    """Generate synthetic tracking data."""
    # State transition matrix (constant velocity model)
    F = torch.tensor([[1.0, dt], [0.0, 1.0]])
    
    # Measurement matrix (we only observe position)
    H = torch.tensor([[1.0, 0.0]])
    
    # Process noise covariance
    G = torch.tensor([[0.5*dt**2], [dt]])
    Q = G @ G.T * process_noise_std**2
    
    # Measurement noise covariance
    R = torch.tensor([[measurement_noise_std**2]])
    
    # Generate true trajectory
    x_true = torch.zeros(time_steps+1, 2)
    x_true[0] = torch.tensor([0.0, 1.0])  # Initial position and velocity
    
    for t in range(1, time_steps+1):
        # Add process noise
        w = torch.randn(1) * process_noise_std
        x_true[t] = F @ x_true[t-1] + G.squeeze() * w
    
    # Generate noisy measurements
    z = torch.zeros(time_steps, 1)
    for t in range(time_steps):
        z[t] = H @ x_true[t+1] + torch.randn(1) * measurement_noise_std
    
    return z, x_true, F, H, Q, R

# Generate data
time_steps = 100
z, x_true, F, H, Q, R = generate_tracking_data(time_steps)

# Initial state and covariance
x0 = torch.tensor([0.0, 1.0])
P0 = torch.eye(2)

# Run Kalman filter
x_filtered, P_filtered = kalman_filter_cholesky(z, x0, P0, F, H, Q, R)

# Visualize the results
plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(2, 1, 1)
plt.plot(range(time_steps+1), x_true[:, 0].numpy(), 'g-', label='True Position')
plt.plot(range(1, time_steps+1), z.squeeze().numpy(), 'ro', alpha=0.5, label='Measurements')
plt.plot(range(time_steps+1), x_filtered[:, 0].numpy(), 'b-', label='Filtered Position')

# Add uncertainty bands (2 standard deviations)
position_std = torch.sqrt(P_filtered[:, 0, 0])
plt.fill_between(range(time_steps+1), 
                 (x_filtered[:, 0] - 2 * position_std).numpy(),
                 (x_filtered[:, 0] + 2 * position_std).numpy(),
                 color='b', alpha=0.2)

plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter: Position Tracking')
plt.legend()
plt.grid(True)

# Velocity plot
plt.subplot(2, 1, 2)
plt.plot(range(time_steps+1), x_true[:, 1].numpy(), 'g-', label='True Velocity')
plt.plot(range(time_steps+1), x_filtered[:, 1].numpy(), 'b-', label='Filtered Velocity')

# Add uncertainty bands (2 standard deviations)
velocity_std = torch.sqrt(P_filtered[:, 1, 1])
plt.fill_between(range(time_steps+1), 
                 (x_filtered[:, 1] - 2 * velocity_std).numpy(),
                 (x_filtered[:, 1] + 2 * velocity_std).numpy(),
                 color='b', alpha=0.2)

plt.xlabel('Time Step')
plt.ylabel('Velocity')
plt.title('Kalman Filter: Velocity Estimation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
# 
# In this notebook, we explored several practical applications of Cholesky decomposition:
# 
# 1. **Solving Linear Systems**: Cholesky decomposition provides an efficient and numerically stable method for solving linear systems when the coefficient matrix is symmetric positive definite.
# 
# 2. **Least Squares Problems**: For overdetermined systems, Cholesky can efficiently solve the normal equations, though QR decomposition may be more stable for ill-conditioned problems.
# 
# 3. **Matrix Inversion**: Cholesky offers a structured approach to inverting symmetric positive definite matrices.
# 
# 4. **Sampling from Multivariate Gaussians**: The decomposition enables efficient generation of correlated random samples.
# 
# 5. **Optimization Algorithms**: In Newton and quasi-Newton methods, Cholesky is used to compute search directions.
# 
# 6. **Kalman Filtering**: Cholesky decomposition helps with numerically stable covariance updates in filtering applications.
# 
# These applications highlight the versatility and importance of Cholesky decomposition in computational mathematics, statistics, optimization, and signal processing.
# 
# The key advantages of Cholesky decomposition in these applications include:
# - Computational efficiency (roughly half the cost of LU decomposition)
# - Numerical stability for well-conditioned problems
# - Natural exploitation of symmetry and positive definiteness
# - Simple and elegant implementation