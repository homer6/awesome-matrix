# %% [markdown]
# # QR Decomposition: Applications
# 
# This notebook explores practical applications of QR decomposition in computational linear algebra and beyond, demonstrating why this factorization is so valuable in various fields.
# 
# We'll investigate the following applications:
# 
# 1. **Solving Linear Systems**
# 2. **Least Squares Problems**
# 3. **QR Algorithm for Eigenvalues**
# 4. **Singular Value Decomposition (SVD)**
# 5. **Applications in Data Science and Machine Learning**
# 
# Each application demonstrates the power and versatility of QR decomposition and provides insights into how this matrix factorization technique is used in practice.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import time
import scipy
import scipy.linalg
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

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
# First, let's implement a reusable QR decomposition function and visualization helpers:

# %%
def qr_decomposition(A, method="householder"):
    """
    Compute QR decomposition of a matrix A.
    
    Args:
        A: Matrix to decompose (numpy array or PyTorch tensor)
        method: 'gram_schmidt', 'modified_gram_schmidt', 'householder', or 'builtin'
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    if method == "gram_schmidt":
        # Classical Gram-Schmidt
        Q = torch.zeros((m, n), dtype=A.dtype)
        R = torch.zeros((n, n), dtype=A.dtype)
        
        for j in range(n):
            v = A[:, j].clone()
            
            for i in range(j):
                R[i, j] = torch.dot(Q[:, i], A[:, j])
                v = v - R[i, j] * Q[:, i]
            
            R[j, j] = torch.norm(v)
            if R[j, j] > 1e-10:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = torch.zeros(m, dtype=A.dtype)
                
        return Q, R
        
    elif method == "modified_gram_schmidt":
        # Modified Gram-Schmidt
        Q = torch.zeros((m, n), dtype=A.dtype)
        R = torch.zeros((n, n), dtype=A.dtype)
        
        U = A.clone()
        
        for i in range(n):
            R[i, i] = torch.norm(U[:, i])
            
            if R[i, i] > 1e-10:
                Q[:, i] = U[:, i] / R[i, i]
            else:
                Q[:, i] = torch.zeros(m, dtype=A.dtype)
            
            for j in range(i+1, n):
                R[i, j] = torch.dot(Q[:, i], U[:, j])
                U[:, j] = U[:, j] - R[i, j] * Q[:, i]
                
        return Q, R
        
    elif method == "householder":
        # Householder reflections
        R = A.clone()
        Q = torch.eye(m, dtype=A.dtype)
        
        for k in range(min(m-1, n)):
            x = R[k:, k]
            
            e1 = torch.zeros_like(x)
            e1[0] = 1.0
            
            alpha = torch.norm(x)
            if x[0] < 0:
                alpha = -alpha
                
            u = x - alpha * e1
            v = u / torch.norm(u)
            
            R[k:, k:] = R[k:, k:] - 2.0 * torch.outer(v, torch.matmul(v, R[k:, k:]))
            Q[:, k:] = Q[:, k:] - 2.0 * torch.matmul(Q[:, k:], torch.outer(v, v))
        
        Q = Q.T
        
        # Ensure positive diagonal for R
        for i in range(min(m, n)):
            if R[i, i] < 0:
                R[i, i:] = -R[i, i:]
                Q[:, i] = -Q[:, i]
                
        return Q, R
        
    elif method == "builtin":
        # Use PyTorch's built-in QR decomposition
        return torch.linalg.qr(A, mode='reduced')
        
    else:
        raise ValueError(f"Unknown method: {method}")

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
# ## 1. Solving Linear Systems using QR Decomposition
# 
# One of the primary applications of QR decomposition is solving linear systems of equations. If we have a system $Ax = b$, we can use QR decomposition to solve it efficiently.
# 
# The process works as follows:
# 1. Decompose $A = QR$
# 2. Rewrite the system as $QRx = b$
# 3. Multiply both sides by $Q^T$ to get $Rx = Q^T b$ (using the fact that $Q^T Q = I$)
# 4. Solve the upper triangular system $Rx = Q^T b$ using back substitution
# 
# Let's implement this approach and compare it with other methods for solving linear systems:

# %%
def solve_with_qr(A, b):
    """Solve a linear system Ax = b using QR decomposition."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float64)
    
    # Perform QR decomposition
    Q, R = qr_decomposition(A, method="householder")
    
    # Compute Q^T b
    y = Q.T @ b
    
    # Solve the upper triangular system Rx = y using back substitution
    n = R.shape[0]
    x = torch.zeros_like(y)
    
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    
    return x

def compare_linear_system_solvers():
    """Compare different methods for solving linear systems."""
    # Create a well-conditioned random matrix
    n = 100
    np.random.seed(42)
    A = np.random.rand(n, n)
    
    # Make it diagonally dominant for better conditioning
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i, :])) + 1.0
    
    # Create a random right-hand side
    b = np.random.rand(n)
    
    # Convert to PyTorch tensors
    A_torch = torch.tensor(A, dtype=torch.float64)
    b_torch = torch.tensor(b, dtype=torch.float64)
    
    # Method 1: Our QR implementation
    start_time = time.time()
    x_qr = solve_with_qr(A_torch, b_torch)
    qr_time = time.time() - start_time
    
    # Method 2: Direct solve using PyTorch
    start_time = time.time()
    x_direct = torch.linalg.solve(A_torch, b_torch)
    direct_time = time.time() - start_time
    
    # Method 3: LU decomposition
    start_time = time.time()
    P, L, U = scipy.linalg.lu(A)
    y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
    x_lu = scipy.linalg.solve_triangular(U, y, lower=False)
    lu_time = time.time() - start_time
    
    # Method 4: SciPy's QR solver
    start_time = time.time()
    Q_scipy, R_scipy = scipy.linalg.qr(A)
    y_scipy = Q_scipy.T @ b
    x_scipy = scipy.linalg.solve_triangular(R_scipy, y_scipy, lower=False)
    scipy_qr_time = time.time() - start_time
    
    # Calculate residuals
    qr_residual = np.linalg.norm(A @ x_qr.numpy() - b) / np.linalg.norm(b)
    direct_residual = np.linalg.norm(A @ x_direct.numpy() - b) / np.linalg.norm(b)
    lu_residual = np.linalg.norm(A @ x_lu - b) / np.linalg.norm(b)
    scipy_qr_residual = np.linalg.norm(A @ x_scipy - b) / np.linalg.norm(b)
    
    # Compare methods
    methods = ["QR (Ours)", "Direct Solve", "LU Decomposition", "QR (SciPy)"]
    times = [qr_time, direct_time, lu_time, scipy_qr_time]
    residuals = [qr_residual, direct_residual, lu_residual, scipy_qr_residual]
    
    # Plot the results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, times)
    plt.ylabel("Time (seconds)")
    plt.title("Solving Linear System Ax = b")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Annotate with time values
    for i, t in enumerate(times):
        plt.text(i, t + 0.001, f"{t:.4f}s", ha='center', va='bottom', fontsize=10)
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, residuals)
    plt.ylabel("Relative Residual")
    plt.title("Solution Accuracy")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Annotate with residual values
    for i, r in enumerate(residuals):
        plt.text(i, r * 1.1, f"{r:.2e}", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("Linear System Solver Comparison:")
    print("-" * 80)
    print(f"{'Method':<20} {'Time (s)':<15} {'Relative Residual':<20} {'Speedup':<10}")
    print("-" * 80)
    
    # Calculate speedup relative to our QR implementation
    reference_time = qr_time
    for method, t, r in zip(methods, times, residuals):
        speedup = reference_time / t
        print(f"{method:<20} {t:<15.6f} {r:<20.2e} {speedup:<10.2f}")
    
    # Return the solutions for further analysis
    return {
        "QR (Ours)": x_qr.numpy(),
        "Direct Solve": x_direct.numpy(),
        "LU Decomposition": x_lu,
        "QR (SciPy)": x_scipy
    }

# Compare different linear system solvers
solution_comparison = compare_linear_system_solvers()

# %% [markdown]
# ### Effect of Matrix Condition Number
# 
# The condition number of a matrix affects the numerical stability of linear system solvers. Let's explore how different methods perform with varying condition numbers:

# %%
def compare_solvers_with_condition_numbers():
    """Compare solver performance with different matrix condition numbers."""
    # Define a range of condition numbers to test
    condition_numbers = [1e1, 1e3, 1e5, 1e7, 1e9]
    
    n = 50  # Matrix size
    
    # Store results
    qr_residuals = []
    lu_residuals = []
    direct_residuals = []
    qr_times = []
    lu_times = []
    direct_times = []
    
    for kappa in condition_numbers:
        # Create a matrix with specified condition number
        # Start with a random orthogonal matrix
        X = np.random.randn(n, n)
        Q, _ = np.linalg.qr(X)
        
        # Create a diagonal matrix with desired condition number
        s1 = 1.0
        sn = s1 / kappa
        S = np.diag(np.linspace(s1, sn, n))
        
        # Form the test matrix
        A = Q @ S @ Q.T
        
        # Create a random right-hand side
        b = np.random.rand(n)
        
        # Convert to PyTorch tensors
        A_torch = torch.tensor(A, dtype=torch.float64)
        b_torch = torch.tensor(b, dtype=torch.float64)
        
        # Method 1: QR decomposition
        start_time = time.time()
        x_qr = solve_with_qr(A_torch, b_torch)
        qr_time = time.time() - start_time
        qr_residual = np.linalg.norm(A @ x_qr.numpy() - b) / np.linalg.norm(b)
        
        # Method 2: LU decomposition
        start_time = time.time()
        try:
            P, L, U = scipy.linalg.lu(A)
            y = scipy.linalg.solve_triangular(L, P @ b, lower=True)
            x_lu = scipy.linalg.solve_triangular(U, y, lower=False)
            lu_residual = np.linalg.norm(A @ x_lu - b) / np.linalg.norm(b)
        except np.linalg.LinAlgError:
            x_lu = np.full_like(b, np.nan)
            lu_residual = np.nan
        lu_time = time.time() - start_time
        
        # Method 3: Direct solve
        start_time = time.time()
        try:
            x_direct = torch.linalg.solve(A_torch, b_torch)
            direct_residual = np.linalg.norm(A @ x_direct.numpy() - b) / np.linalg.norm(b)
        except RuntimeError:
            x_direct = torch.full_like(b_torch, float('nan'))
            direct_residual = float('nan')
        direct_time = time.time() - start_time
        
        # Store results
        qr_residuals.append(qr_residual)
        lu_residuals.append(lu_residual)
        direct_residuals.append(direct_residual)
        qr_times.append(qr_time)
        lu_times.append(lu_time)
        direct_times.append(direct_time)
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Residuals
    plt.subplot(2, 1, 1)
    plt.loglog(condition_numbers, qr_residuals, 'o-', label='QR')
    plt.loglog(condition_numbers, lu_residuals, 's-', label='LU')
    plt.loglog(condition_numbers, direct_residuals, '^-', label='Direct')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Condition Number")
    plt.ylabel("Relative Residual")
    plt.title("Solver Accuracy vs. Matrix Condition Number")
    plt.legend()
    
    # Computation time
    plt.subplot(2, 1, 2)
    plt.semilogx(condition_numbers, qr_times, 'o-', label='QR')
    plt.semilogx(condition_numbers, lu_times, 's-', label='LU')
    plt.semilogx(condition_numbers, direct_times, '^-', label='Direct')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Condition Number")
    plt.ylabel("Time (seconds)")
    plt.title("Solver Performance vs. Matrix Condition Number")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("Solver Performance with Different Condition Numbers:")
    print("-" * 100)
    print(f"{'Condition Number':<20} {'QR Residual':<15} {'LU Residual':<15} {'Direct Residual':<15}")
    print("-" * 100)
    
    for kappa, qr_res, lu_res, direct_res in zip(condition_numbers, qr_residuals, lu_residuals, direct_residuals):
        print(f"{kappa:<20.1e} {qr_res:<15.2e} {lu_res:<15.2e} {direct_res:<15.2e}")
    
    return condition_numbers, qr_residuals, lu_residuals, direct_residuals

# Compare solvers with different condition numbers
cond_numbers, qr_res, lu_res, direct_res = compare_solvers_with_condition_numbers()

# %% [markdown]
# ## 2. Least Squares Problems
# 
# QR decomposition is particularly well-suited for solving least squares problems. Given an overdetermined system (more equations than unknowns), we want to find $x$ that minimizes $||Ax - b||_2$.
# 
# The solution is given by the normal equations: $A^T A x = A^T b$. However, forming $A^T A$ explicitly can lead to numerical issues. Instead, we can use QR decomposition:
# 
# 1. Decompose $A = QR$
# 2. The least squares solution is then $x = R^{-1} Q^T b$, or equivalently, $Rx = Q^T b$
# 
# Let's implement and demonstrate this approach:

# %%
def solve_least_squares_with_qr(A, b):
    """Solve a least squares problem using QR decomposition."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b, dtype=torch.float64)
    
    # Perform QR decomposition
    Q, R = qr_decomposition(A, method="householder")
    
    # Compute Q^T b
    y = Q.T @ b
    
    # Solve the upper triangular system Rx = y
    n = R.shape[1]  # Number of columns
    x = torch.zeros(n, dtype=A.dtype)
    
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    
    return x

def demonstrate_least_squares():
    """Demonstrate solving a least squares problem with QR decomposition."""
    # Create a simple linear regression problem
    np.random.seed(42)
    n = 100  # Number of data points
    p = 3    # Number of parameters (including intercept)
    
    # Create design matrix (features)
    X = np.random.rand(n, p-1)  # Random features
    X = np.column_stack([np.ones(n), X])  # Add intercept column
    
    # True parameters
    beta_true = np.array([2.5, -1.8, 3.7])
    
    # Generate noisy observations
    noise_level = 0.5
    y = X @ beta_true + noise_level * np.random.randn(n)
    
    # Solve using different methods
    methods = []
    params = []
    residuals = []
    times = []
    
    # Method 1: Our QR implementation
    start_time = time.time()
    beta_qr = solve_least_squares_with_qr(X, y)
    qr_time = time.time() - start_time
    y_pred_qr = X @ beta_qr.numpy()
    qr_residual = np.linalg.norm(y - y_pred_qr) / np.linalg.norm(y)
    
    methods.append("QR (Ours)")
    params.append(beta_qr.numpy())
    residuals.append(qr_residual)
    times.append(qr_time)
    
    # Method 2: Normal equations
    start_time = time.time()
    beta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
    normal_time = time.time() - start_time
    y_pred_normal = X @ beta_normal
    normal_residual = np.linalg.norm(y - y_pred_normal) / np.linalg.norm(y)
    
    methods.append("Normal Equations")
    params.append(beta_normal)
    residuals.append(normal_residual)
    times.append(normal_time)
    
    # Method 3: SciPy's QR solver
    start_time = time.time()
    Q_scipy, R_scipy = scipy.linalg.qr(X)
    z = Q_scipy.T @ y
    beta_scipy = scipy.linalg.solve_triangular(R_scipy[:p, :], z[:p], lower=False)
    scipy_qr_time = time.time() - start_time
    y_pred_scipy = X @ beta_scipy
    scipy_qr_residual = np.linalg.norm(y - y_pred_scipy) / np.linalg.norm(y)
    
    methods.append("QR (SciPy)")
    params.append(beta_scipy)
    residuals.append(scipy_qr_residual)
    times.append(scipy_qr_time)
    
    # Method 4: NumPy's least squares solver
    start_time = time.time()
    beta_np, residuals_np, rank_np, s_np = np.linalg.lstsq(X, y, rcond=None)
    np_time = time.time() - start_time
    y_pred_np = X @ beta_np
    np_residual = np.linalg.norm(y - y_pred_np) / np.linalg.norm(y)
    
    methods.append("np.linalg.lstsq")
    params.append(beta_np)
    residuals.append(np_residual)
    times.append(np_time)
    
    # Plot the results
    # Compare parameters
    plt.figure(figsize=(15, 12))
    
    plt.subplot(2, 2, 1)
    param_names = ["Intercept", "Coefficient 1", "Coefficient 2"]
    x_pos = np.arange(len(param_names))
    width = 0.2
    
    # Plot bars for each method
    for i, (method, beta) in enumerate(zip(methods, params)):
        plt.bar(x_pos + i*width - 0.3, beta, width, label=method)
    
    # Add true parameter values
    plt.plot(x_pos, beta_true, 'ro', markersize=8, label='True Values')
    
    plt.xticks(x_pos, param_names)
    plt.ylabel("Parameter Value")
    plt.title("Estimated Parameters")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot computation time
    plt.subplot(2, 2, 2)
    plt.bar(methods, times)
    plt.ylabel("Time (seconds)")
    plt.title("Computation Time")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Annotate with time values
    for i, t in enumerate(times):
        plt.text(i, t + 0.0001, f"{t:.6f}s", ha='center', va='bottom', fontsize=10)
    
    # Plot residuals
    plt.subplot(2, 2, 3)
    plt.bar(methods, residuals)
    plt.ylabel("Relative Residual")
    plt.title("Solution Accuracy")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Scatter plot of actual vs. predicted
    plt.subplot(2, 2, 4)
    plt.scatter(y, y_pred_qr, alpha=0.5, label='Predictions')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Fit')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Actual vs. Predicted (QR Method)\nR² = {1 - qr_residual**2:.4f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print parameter estimates
    print("Least Squares Parameter Estimates:")
    print("-" * 80)
    print(f"{'Parameter':<15} {'True Value':<15} " + " ".join([f"{method:<15}" for method in methods]))
    print("-" * 80)
    
    for i, name in enumerate(param_names):
        param_values = [p[i] for p in params]
        print(f"{name:<15} {beta_true[i]:<15.4f} " + " ".join([f"{val:<15.4f}" for val in param_values]))
    
    print("\nRelative Residuals:")
    print("-" * 50)
    for method, residual in zip(methods, residuals):
        print(f"{method:<20} {residual:<15.2e}")
    
    return X, y, params

# Demonstrate least squares regression
X_data, y_data, beta_estimates = demonstrate_least_squares()

# %% [markdown]
# ### Geometric Interpretation of Least Squares with QR
# 
# Let's visualize how QR decomposition helps in solving least squares problems geometrically. In particular, we'll see how the projection of $b$ onto the column space of $A$ gives us the least squares solution:

# %%
def visualize_least_squares_2d():
    """Visualize least squares solution in 2D using QR decomposition."""
    # Create a simple 2D problem
    np.random.seed(42)
    n = 20  # Number of data points
    
    # Create a single feature and add intercept
    x = np.random.rand(n)
    X = np.column_stack([np.ones(n), x])
    
    # True parameters and noisy observations
    beta_true = np.array([2.0, 1.5])
    noise_level = 0.3
    y = X @ beta_true + noise_level * np.random.randn(n)
    
    # Solve using QR decomposition
    Q, R = scipy.linalg.qr(X)
    z = Q.T @ y
    beta_qr = scipy.linalg.solve_triangular(R[:2, :], z[:2], lower=False)
    y_pred = X @ beta_qr
    
    # Calculate projections
    projection = Q[:, :2] @ z[:2]  # Projection of y onto col(X)
    residual_vector = y - projection  # Orthogonal to col(X)
    
    # Create scatter plot of data and regression line
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, label='Data Points')
    
    # Plot the regression line
    x_line = np.linspace(0, 1, 100)
    y_line = beta_qr[0] + beta_qr[1] * x_line
    plt.plot(x_line, y_line, 'r-', label='Regression Line')
    
    # Plot residuals as vertical lines
    for i in range(n):
        plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--', alpha=0.5)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Least Squares Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Now visualize in the column space of X and its orthogonal complement
    plt.figure(figsize=(12, 10))
    
    # Project data into the Q space (transformed coordinates)
    y_Q = Q.T @ y
    
    # In Q space, the first two coordinates represent the column space of X
    # and the remaining coordinates represent the orthogonal complement
    
    # Plot in the column space of X (first two dimensions of Q)
    plt.subplot(2, 2, 1)
    plt.scatter([0], [0], color='black', s=50, marker='o', label='Origin')
    plt.scatter([0], [y_Q[0]], color='blue', s=50, marker='s', label='y_Q[0]')
    plt.scatter([y_Q[1]], [0], color='green', s=50, marker='^', label='y_Q[1]')
    plt.scatter([y_Q[1]], [y_Q[0]], color='red', s=100, marker='*', label='Projection')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('Q₁')
    plt.ylabel('Q₂')
    plt.title('Projection in Column Space of X')
    plt.legend()
    
    # Plot original vector and its projection
    plt.subplot(2, 2, 2)
    
    # We'll use the first 3 dimensions for visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the original vector y in the Q-basis
    ax.quiver(0, 0, 0, y_Q[0], y_Q[1], y_Q[2], color='blue', label='y')
    
    # Plot the projection onto the first two dimensions
    ax.quiver(0, 0, 0, y_Q[0], y_Q[1], 0, color='red', label='Projection')
    
    # Plot the residual (orthogonal to the column space)
    ax.quiver(y_Q[0], y_Q[1], 0, 0, 0, y_Q[2], color='green', label='Residual')
    
    # Plot the coordinate axes
    length = max(abs(y_Q[:3])) * 1.2
    ax.quiver(0, 0, 0, length, 0, 0, color='k', alpha=0.5)
    ax.quiver(0, 0, 0, 0, length, 0, color='k', alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, length, color='k', alpha=0.5)
    
    ax.set_xlabel('Q₁')
    ax.set_ylabel('Q₂')
    ax.set_zlabel('Q₃')
    ax.set_title('Decomposition of y in Q-basis')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("Least Squares Solution:")
    print(f"True parameters: {beta_true}")
    print(f"QR solution: {beta_qr}")
    print(f"Residual norm: {np.linalg.norm(y - y_pred):.4f}")
    print(f"Q^T y: {y_Q[:5]} (first 5 components)")
    print(f"Norm of residual components: {np.linalg.norm(y_Q[2:]):.4f}")
    
    return X, y, beta_qr, Q, R

# Visualize least squares solution in 2D
X_2d, y_2d, beta_2d, Q_2d, R_2d = visualize_least_squares_2d()

# %% [markdown]
# ## 3. QR Algorithm for Computing Eigenvalues
# 
# The QR algorithm is one of the most important methods for computing eigenvalues of matrices. It's an iterative method that works as follows:
# 
# 1. Start with a matrix $A_0 = A$
# 2. For $k = 0, 1, 2, ...$:
#    - Compute the QR decomposition: $A_k = Q_k R_k$
#    - Form the next iteration: $A_{k+1} = R_k Q_k$
# 3. As $k$ increases, $A_k$ converges to a Schur form (upper triangular or block upper triangular with 2×2 blocks on the diagonal for real matrices)
# 
# Let's implement and visualize this algorithm:

# %%
def qr_algorithm_for_eigenvalues(A, max_iter=100, tol=1e-8):
    """
    Compute eigenvalues using the basic QR algorithm.
    
    Args:
        A: Input matrix (torch tensor or numpy array)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        eigenvalues: Estimated eigenvalues
        iterations: Number of iterations performed
        A_final: Final matrix after iterations
        convergence: List of convergence metrics per iteration
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    A_k = A.clone()
    
    # For tracking convergence
    convergence = []
    
    # Store matrices for visualization
    matrices = [A_k.clone()]
    
    for k in range(max_iter):
        # Compute QR decomposition
        Q, R = torch.linalg.qr(A_k)
        
        # Form the next iteration
        A_next = R @ Q
        
        # Track convergence (sum of off-diagonal elements)
        off_diag_sum = torch.sum(torch.abs(A_next - torch.diag(torch.diag(A_next)))).item()
        convergence.append(off_diag_sum)
        
        # Store the matrix
        matrices.append(A_next.clone())
        
        # Check for convergence
        if off_diag_sum < tol:
            break
            
        A_k = A_next
    
    # Extract eigenvalues from the diagonal
    eigenvalues = torch.diag(A_k)
    
    return eigenvalues, k+1, A_k, convergence, matrices

def demonstrate_qr_algorithm():
    """Demonstrate the QR algorithm for eigenvalue computation."""
    # Create a matrix with known eigenvalues
    n = 5
    np.random.seed(42)
    
    # Start with diagonal matrix with known eigenvalues
    true_eigenvalues = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    D = np.diag(true_eigenvalues)
    
    # Create a random orthogonal matrix
    X = np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    
    # Form a matrix with the same eigenvalues
    A = Q @ D @ Q.T
    
    # Apply QR algorithm
    computed_eigs, iterations, A_final, convergence, matrices = qr_algorithm_for_eigenvalues(A)
    
    # Also compute eigenvalues with NumPy for comparison
    np_eigs = np.linalg.eigvals(A)
    
    # Sort eigenvalues for comparison
    true_eigenvalues = np.sort(true_eigenvalues)
    computed_eigs = torch.sort(computed_eigs)[0].numpy()
    np_eigs = np.sort(np_eigs)
    
    # Visualize convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(convergence, 'o-')
    plt.xlabel("Iteration")
    plt.ylabel("Off-diagonal Sum")
    plt.title("QR Algorithm Convergence")
    plt.grid(True, alpha=0.3)
    
    # Compare eigenvalues
    plt.subplot(1, 2, 2)
    width = 0.25
    x = np.arange(n)
    
    plt.bar(x - width, true_eigenvalues, width, label='True')
    plt.bar(x, computed_eigs, width, label='QR Algorithm')
    plt.bar(x + width, np_eigs.real, width, label='NumPy')
    
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Comparison")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize matrix evolution
    # Show a few selected iterations
    num_matrices = len(matrices)
    iterations_to_show = [0, 1, 2, 5, 10, num_matrices-1]
    iterations_to_show = [min(i, num_matrices-1) for i in iterations_to_show]
    
    plt.figure(figsize=(15, 8))
    
    for i, iter_idx in enumerate(iterations_to_show):
        plt.subplot(2, 3, i+1)
        matrix = matrices[iter_idx].numpy()
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title(f"Iteration {iter_idx}")
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("QR Algorithm for Eigenvalues:")
    print(f"Number of iterations: {iterations}")
    print("-" * 60)
    print(f"{'Index':<10} {'True':<15} {'QR Algorithm':<15} {'NumPy':<15} {'Error':<15}")
    print("-" * 60)
    
    for i in range(n):
        qr_error = abs(computed_eigs[i] - true_eigenvalues[i])
        print(f"{i:<10} {true_eigenvalues[i]:<15.6f} {computed_eigs[i]:<15.6f} {np_eigs[i].real:<15.6f} {qr_error:<15.2e}")
    
    return A, true_eigenvalues, computed_eigs, matrices

# Demonstrate QR algorithm for eigenvalues
A_eigen, true_eigs, qr_eigs, matrix_sequence = demonstrate_qr_algorithm()

# %% [markdown]
# ### Shifted QR Algorithm
# 
# The basic QR algorithm can be slow to converge, especially for matrices with clustered eigenvalues. A common enhancement is the "shifted QR algorithm", which accelerates convergence by incorporating shifts:
# 
# 1. Start with a matrix $A_0 = A$
# 2. For $k = 0, 1, 2, ...$:
#    - Choose a shift $\mu_k$ (often the bottom-right element of $A_k$)
#    - Compute the QR decomposition: $A_k - \mu_k I = Q_k R_k$
#    - Form the next iteration: $A_{k+1} = R_k Q_k + \mu_k I$
# 
# Let's implement and compare this with the basic QR algorithm:

# %%
def shifted_qr_algorithm(A, max_iter=100, tol=1e-8):
    """
    Compute eigenvalues using the shifted QR algorithm.
    
    Args:
        A: Input matrix (torch tensor or numpy array)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        eigenvalues: Estimated eigenvalues
        iterations: Number of iterations performed
        A_final: Final matrix after iterations
        convergence: List of convergence metrics per iteration
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    n = A.shape[0]
    A_k = A.clone()
    
    # For tracking convergence
    convergence = []
    
    # Store matrices for visualization
    matrices = [A_k.clone()]
    
    for k in range(max_iter):
        # Choose a shift (bottom-right element)
        mu = A_k[-1, -1].item()
        
        # Compute shifted QR decomposition
        A_shifted = A_k - mu * torch.eye(n, dtype=A.dtype)
        Q, R = torch.linalg.qr(A_shifted)
        
        # Form the next iteration
        A_next = R @ Q + mu * torch.eye(n, dtype=A.dtype)
        
        # Track convergence (sum of off-diagonal elements)
        off_diag_sum = torch.sum(torch.abs(A_next - torch.diag(torch.diag(A_next)))).item()
        convergence.append(off_diag_sum)
        
        # Store the matrix
        matrices.append(A_next.clone())
        
        # Check for convergence
        if off_diag_sum < tol:
            break
            
        A_k = A_next
    
    # Extract eigenvalues from the diagonal
    eigenvalues = torch.diag(A_k)
    
    return eigenvalues, k+1, A_k, convergence, matrices

def compare_qr_algorithms():
    """Compare basic and shifted QR algorithms."""
    # Create a matrix with clustered eigenvalues
    n = 5
    np.random.seed(42)
    
    # Clustered eigenvalues
    true_eigenvalues = np.array([1.0, 1.1, 3.0, 3.1, 5.0])
    D = np.diag(true_eigenvalues)
    
    # Create a random orthogonal matrix
    X = np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    
    # Form a matrix with the same eigenvalues
    A = Q @ D @ Q.T
    
    # Apply basic QR algorithm
    basic_eigs, basic_iters, _, basic_conv, _ = qr_algorithm_for_eigenvalues(A)
    
    # Apply shifted QR algorithm
    shifted_eigs, shifted_iters, _, shifted_conv, _ = shifted_qr_algorithm(A)
    
    # Also compute eigenvalues with NumPy for comparison
    np_eigs = np.linalg.eigvals(A)
    
    # Sort eigenvalues for comparison
    true_eigenvalues = np.sort(true_eigenvalues)
    basic_eigs = torch.sort(basic_eigs)[0].numpy()
    shifted_eigs = torch.sort(shifted_eigs)[0].numpy()
    np_eigs = np.sort(np_eigs)
    
    # Visualize convergence
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.semilogy(basic_conv, 'o-', label='Basic QR')
    plt.semilogy(shifted_conv, 's-', label='Shifted QR')
    plt.xlabel("Iteration")
    plt.ylabel("Off-diagonal Sum")
    plt.title("Convergence Comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Compare eigenvalues
    plt.subplot(2, 2, 2)
    width = 0.2
    x = np.arange(n)
    
    plt.bar(x - 1.5*width, true_eigenvalues, width, label='True')
    plt.bar(x - 0.5*width, basic_eigs, width, label='Basic QR')
    plt.bar(x + 0.5*width, shifted_eigs, width, label='Shifted QR')
    plt.bar(x + 1.5*width, np_eigs.real, width, label='NumPy')
    
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Comparison")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot errors
    plt.subplot(2, 2, 3)
    basic_errors = [abs(basic_eigs[i] - true_eigenvalues[i]) for i in range(n)]
    shifted_errors = [abs(shifted_eigs[i] - true_eigenvalues[i]) for i in range(n)]
    numpy_errors = [abs(np_eigs[i].real - true_eigenvalues[i]) for i in range(n)]
    
    plt.bar(x - width, basic_errors, width, label='Basic QR')
    plt.bar(x, shifted_errors, width, label='Shifted QR')
    plt.bar(x + width, numpy_errors, width, label='NumPy')
    
    plt.xlabel("Index")
    plt.ylabel("Absolute Error")
    plt.title("Eigenvalue Error Comparison")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Iterations and timing
    plt.subplot(2, 2, 4)
    plt.bar(["Basic QR", "Shifted QR"], [basic_iters, shifted_iters])
    plt.ylabel("Number of Iterations")
    plt.title("Convergence Speed")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("QR Algorithm Comparison:")
    print(f"Basic QR iterations: {basic_iters}")
    print(f"Shifted QR iterations: {shifted_iters}")
    print("-" * 100)
    print(f"{'Index':<6} {'True':<10} {'Basic QR':<15} {'Shifted QR':<15} {'NumPy':<15} {'Basic Error':<15} {'Shifted Error':<15}")
    print("-" * 100)
    
    for i in range(n):
        basic_error = abs(basic_eigs[i] - true_eigenvalues[i])
        shifted_error = abs(shifted_eigs[i] - true_eigenvalues[i])
        print(f"{i:<6} {true_eigenvalues[i]:<10.6f} {basic_eigs[i]:<15.6f} {shifted_eigs[i]:<15.6f} {np_eigs[i].real:<15.6f} {basic_error:<15.2e} {shifted_error:<15.2e}")
    
    return A, true_eigenvalues, basic_eigs, shifted_eigs

# Compare basic and shifted QR algorithms
A_clustered, true_clustered_eigs, basic_eigs_clustered, shifted_eigs_clustered = compare_qr_algorithms()

# %% [markdown]
# ## 4. Computing the Singular Value Decomposition (SVD) via QR
# 
# The Singular Value Decomposition (SVD) is another fundamental matrix factorization that can be computed using QR decomposition. One approach is to first compute the eigendecomposition of $A^T A$ or $AA^T$ using the QR algorithm, and then derive the SVD from there.
# 
# For a matrix $A \in \mathbb{R}^{m \times n}$, the SVD is $A = USV^T$, where:
# - $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix
# - $S \in \mathbb{R}^{m \times n}$ is a diagonal matrix of singular values
# - $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix
# 
# Let's implement a simplified version of this algorithm:

# %%
def svd_via_qr(A, max_iter=100, tol=1e-8):
    """
    Compute the SVD using QR algorithm for eigendecomposition.
    
    Args:
        A: Input matrix (torch tensor or numpy array)
        max_iter: Maximum number of iterations for QR algorithm
        tol: Convergence tolerance
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    # Compute A^T A and AA^T
    ATA = A.T @ A
    AAT = A @ A.T
    
    # Find eigenvectors of A^T A (right singular vectors)
    ATA_eigs, ATA_iters, ATA_final, _, _ = shifted_qr_algorithm(ATA, max_iter, tol)
    
    # Extract V and singular values
    # Note: This simplified version assumes ATA_final is diagonal (may not be exactly true)
    S = torch.sqrt(torch.abs(torch.diag(ATA_final)))
    
    # Sort singular values in descending order
    S_sorted, indices = torch.sort(S, descending=True)
    
    # Compute V via eigenvectors of A^T A
    # In practice, we'd need to use a more refined approach to get V exactly
    # This is a simplified approximation
    V = torch.zeros((n, n), dtype=A.dtype)
    for i, idx in enumerate(indices):
        v = torch.zeros(n, dtype=A.dtype)
        v[idx] = 1.0
        
        # Apply inverse iteration to get a better eigenvector
        for _ in range(5):
            v = torch.linalg.solve(ATA - ATA_final[idx, idx] * torch.eye(n, dtype=A.dtype) + 1e-10 * torch.eye(n, dtype=A.dtype), v)
            v = v / torch.norm(v)
        
        V[:, i] = v
    
    # Compute U via the relation A = USV^T
    U = torch.zeros((m, m), dtype=A.dtype)
    for i in range(min(m, n)):
        if S_sorted[i] > 1e-10:
            U[:, i] = (A @ V[:, i]) / S_sorted[i]
    
    # Complete U to be orthogonal if m > n
    if m > n:
        # Find a basis for the nullspace of A^T
        Q, _ = torch.linalg.qr(U[:, :n], mode='complete')
        U = Q
    
    return U, S_sorted, V

def demonstrate_svd():
    """Demonstrate SVD computation using QR algorithm."""
    # Create a matrix for SVD
    m, n = 5, 3  # More rows than columns
    np.random.seed(42)
    
    # Create a matrix with known singular values
    true_singular_values = np.array([10.0, 5.0, 1.0])
    
    # Create random orthogonal matrices
    U_true = scipy.linalg.orth(np.random.randn(m, m))
    V_true = scipy.linalg.orth(np.random.randn(n, n))
    
    # Create S
    S_true = np.zeros((m, n))
    for i in range(min(m, n)):
        S_true[i, i] = true_singular_values[i]
    
    # Form the matrix
    A = U_true @ S_true @ V_true.T
    
    # Compute SVD using our QR-based method
    U, S, V = svd_via_qr(A)
    
    # Also compute SVD with NumPy for comparison
    U_np, S_np, VT_np = np.linalg.svd(A, full_matrices=True)
    
    # Visualize the results
    plt.figure(figsize=(12, 8))
    
    # Compare singular values
    plt.subplot(2, 2, 1)
    width = 0.3
    x = np.arange(min(m, n))
    
    plt.bar(x - width/2, true_singular_values, width, label='True')
    plt.bar(x + width/2, S.numpy()[:min(m, n)], width, label='QR-based SVD')
    
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Value Comparison")
    plt.xticks(x)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show the original matrix and its reconstruction
    plt.subplot(2, 2, 2)
    S_diag = torch.zeros((m, n), dtype=A.dtype)
    for i in range(min(m, n)):
        S_diag[i, i] = S[i]
    A_reconstructed = U @ S_diag @ V.T
    
    reconstruction_error = torch.norm(torch.tensor(A, dtype=torch.float64) - A_reconstructed).item()
    
    sns.heatmap(A, annot=True, fmt=".2f", cmap=blue_cmap)
    plt.title("Original Matrix A")
    
    plt.subplot(2, 2, 3)
    sns.heatmap(A_reconstructed.numpy(), annot=True, fmt=".2f", cmap=blue_cmap)
    plt.title(f"Reconstructed A = USV^T\nError: {reconstruction_error:.2e}")
    
    # Orthogonality check for U and V
    plt.subplot(2, 2, 4)
    UTU = U.T @ U
    VTV = V.T @ V
    
    ortho_error_U = torch.norm(UTU - torch.eye(m, dtype=A.dtype)).item()
    ortho_error_V = torch.norm(VTV - torch.eye(n, dtype=A.dtype)).item()
    
    plt.bar(["U^T U - I", "V^T V - I"], [ortho_error_U, ortho_error_V])
    plt.ylabel("Frobenius Norm")
    plt.title("Orthogonality Check")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("SVD via QR Algorithm:")
    print("-" * 80)
    print(f"{'Index':<10} {'True S':<15} {'QR-based S':<15} {'NumPy S':<15} {'Error':<15}")
    print("-" * 80)
    
    for i in range(min(m, n)):
        error = abs(S.numpy()[i] - true_singular_values[i])
        print(f"{i:<10} {true_singular_values[i]:<15.6f} {S.numpy()[i]:<15.6f} {S_np[i]:<15.6f} {error:<15.2e}")
    
    print(f"\nReconstruction error: {reconstruction_error:.2e}")
    print(f"Orthogonality error (U): {ortho_error_U:.2e}")
    print(f"Orthogonality error (V): {ortho_error_V:.2e}")
    
    return A, U, S, V, true_singular_values

# Demonstrate SVD computation using QR algorithm
A_svd, U_svd, S_svd, V_svd, true_S = demonstrate_svd()

# %% [markdown]
# ## 5. Applications in Data Science and Machine Learning
# 
# QR decomposition has numerous applications in data science and machine learning. Let's explore a few of these applications.
# 
# ### 5.1 Principal Component Analysis (PCA) using QR
# 
# PCA is typically performed using SVD, but it can also be computed using QR decomposition with column pivoting. Let's implement and demonstrate this approach:

# %%
def qr_with_column_pivoting(A):
    """Compute QR decomposition with column pivoting."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    P = torch.eye(n, dtype=A.dtype)
    
    # Use SciPy's implementation for simplicity
    Q, R, P_indices = scipy.linalg.qr(A.numpy(), pivoting=True)
    
    # Convert P indices to a permutation matrix
    P_matrix = torch.zeros((n, n), dtype=A.dtype)
    for i, p in enumerate(P_indices):
        P_matrix[p, i] = 1.0
    
    return torch.tensor(Q), torch.tensor(R), P_matrix

def pca_with_qr(X, k=2):
    """
    Perform PCA using QR decomposition with column pivoting.
    
    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of principal components
        
    Returns:
        principal_components: Top k principal components
        projected_data: Data projected onto principal components
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64)
    
    # Center the data
    X_centered = X - X.mean(dim=0)
    
    # Compute the covariance matrix
    n = X.shape[0]
    cov = (X_centered.T @ X_centered) / (n - 1)
    
    # Use QR with column pivoting on the covariance matrix
    Q, R, P = qr_with_column_pivoting(cov)
    
    # The first k columns of Q*P are approximations of the principal components
    principal_components = (Q @ P)[:, :k]
    
    # Project the data onto the principal components
    projected_data = X_centered @ principal_components
    
    return principal_components, projected_data

def demonstrate_pca():
    """Demonstrate PCA using QR decomposition."""
    # Generate a 2D dataset with correlation
    np.random.seed(42)
    n_samples = 200
    
    # Create correlated features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n_samples)
    
    # Add some noise in a different direction
    x3 = 0.1 * x1 + 0.1 * x2 + 0.2 * np.random.normal(0, 1, n_samples)
    
    # Create a data matrix
    X = np.column_stack([x1, x2, x3])
    
    # Perform PCA using our QR-based method
    n_components = 2
    principal_components, projected_data = pca_with_qr(X, k=n_components)
    
    # Also compute PCA with sklearn for comparison
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    projected_data_sklearn = pca.fit_transform(X)
    
    # Visualize the results
    plt.figure(figsize=(15, 8))
    
    # Original data (first two dimensions)
    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Original Data (First 2 Dimensions)")
    plt.grid(True, alpha=0.3)
    
    # Original data in 3D
    ax = plt.subplot(2, 2, 2, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.5)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    ax.set_title("Original Data (3D)")
    
    # Projected data (QR-based PCA)
    plt.subplot(2, 2, 3)
    plt.scatter(projected_data.numpy()[:, 0], projected_data.numpy()[:, 1], alpha=0.5)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("QR-based PCA")
    plt.grid(True, alpha=0.3)
    
    # Projected data (sklearn PCA)
    plt.subplot(2, 2, 4)
    plt.scatter(projected_data_sklearn[:, 0], projected_data_sklearn[:, 1], alpha=0.5)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("sklearn PCA")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare principal components
    print("QR-based Principal Components:")
    print(principal_components.numpy())
    print("\nsklearn Principal Components:")
    print(pca.components_.T)
    
    # Note: PCA components may differ in sign but should span the same space
    
    return X, principal_components, projected_data

# Demonstrate PCA using QR decomposition
X_pca, pc_qr, projected_qr = demonstrate_pca()

# %% [markdown]
# ### 5.2 QR for Feature Selection
# 
# QR decomposition with column pivoting can also be used for feature selection. The idea is that the pivoting strategy selects the most linearly independent columns, which can be used to identify the most important features:

# %%
def feature_selection_with_qr(X, k):
    """
    Select k most important features using QR decomposition with column pivoting.
    
    Args:
        X: Data matrix (n_samples, n_features)
        k: Number of features to select
        
    Returns:
        selected_indices: Indices of selected features
        X_selected: Data with only selected features
    """
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float64)
    
    # Center and scale the data
    X_centered = X - X.mean(dim=0)
    X_scaled = X_centered / X_centered.std(dim=0)
    
    # Compute QR decomposition with column pivoting
    Q, R, P = scipy.linalg.qr(X_scaled.numpy(), pivoting=True)
    
    # The first k columns selected by the pivoting are the most important
    selected_indices = P[:k]
    X_selected = X[:, selected_indices]
    
    return selected_indices, X_selected

def demonstrate_feature_selection():
    """Demonstrate feature selection using QR decomposition."""
    # Generate a dataset with some informative and some noise features
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    # Create a few important features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    
    # Target is a function of the important features
    y = 2*x1 + 0.5*x2 - 1.5*x3 + 0.1*np.random.normal(0, 1, n_samples)
    
    # Create noise features
    X_noise = np.random.normal(0, 1, (n_samples, n_features-3))
    
    # Combine important and noise features
    X = np.column_stack([x1, x2, x3, X_noise])
    
    # Create feature names
    feature_names = ["Important 1", "Important 2", "Important 3"] + [f"Noise {i+1}" for i in range(n_features-3)]
    
    # Select features using QR decomposition
    k = 5  # Number of features to select
    selected_indices, X_selected = feature_selection_with_qr(X, k)
    
    # For comparison, use a linear model with L1 regularization (Lasso)
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_scaled, y)
    
    # Get feature importance from Lasso
    lasso_importance = np.abs(lasso.coef_)
    lasso_selected = np.argsort(lasso_importance)[::-1][:k]
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    
    # Show which features were selected
    plt.subplot(1, 2, 1)
    plt.bar(range(n_features), np.zeros(n_features), alpha=0.1)  # Placeholder bars
    plt.bar(selected_indices, np.ones(k), alpha=0.7, label='Selected by QR')
    
    # Highlight the truly important features
    for i in range(3):
        plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
    
    plt.xlabel("Feature Index")
    plt.title(f"QR-based Feature Selection\nSelected: {', '.join([feature_names[i] for i in selected_indices])}")
    plt.xticks(range(n_features), feature_names, rotation=90)
    plt.grid(True, alpha=0.3)
    
    # Show Lasso feature importance
    plt.subplot(1, 2, 2)
    sorted_idx = np.argsort(lasso_importance)[::-1]
    plt.bar(range(n_features), lasso_importance[sorted_idx])
    
    plt.xlabel("Feature")
    plt.title("Lasso Feature Importance")
    plt.xticks(range(n_features), [feature_names[i] for i in sorted_idx], rotation=90)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate feature selection by training a simple model
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train on all features
    reg_all = LinearRegression()
    reg_all.fit(X_train, y_train)
    y_pred_all = reg_all.predict(X_test)
    rmse_all = np.sqrt(mean_squared_error(y_test, y_pred_all))
    r2_all = r2_score(y_test, y_pred_all)
    
    # Train on QR-selected features
    reg_qr = LinearRegression()
    reg_qr.fit(X_train[:, selected_indices], y_train)
    y_pred_qr = reg_qr.predict(X_test[:, selected_indices])
    rmse_qr = np.sqrt(mean_squared_error(y_test, y_pred_qr))
    r2_qr = r2_score(y_test, y_pred_qr)
    
    # Train on Lasso-selected features
    reg_lasso = LinearRegression()
    reg_lasso.fit(X_train[:, lasso_selected], y_train)
    y_pred_lasso = reg_lasso.predict(X_test[:, lasso_selected])
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    r2_lasso = r2_score(y_test, y_pred_lasso)
    
    # Train on only the true important features
    reg_true = LinearRegression()
    reg_true.fit(X_train[:, :3], y_train)
    y_pred_true = reg_true.predict(X_test[:, :3])
    rmse_true = np.sqrt(mean_squared_error(y_test, y_pred_true))
    r2_true = r2_score(y_test, y_pred_true)
    
    # Print the results
    print("Feature Selection Results:")
    print("-" * 80)
    print(f"{'Method':<20} {'Features':<40} {'RMSE':<10} {'R²':<10}")
    print("-" * 80)
    
    print(f"{'All Features':<20} {'All 10 features':<40} {rmse_all:<10.4f} {r2_all:<10.4f}")
    print(f"{'QR Selection':<20} {', '.join([feature_names[i] for i in selected_indices]):<40} {rmse_qr:<10.4f} {r2_qr:<10.4f}")
    print(f"{'Lasso Selection':<20} {', '.join([feature_names[i] for i in lasso_selected]):<40} {rmse_lasso:<10.4f} {r2_lasso:<10.4f}")
    print(f"{'True Important':<20} {'Important 1, Important 2, Important 3':<40} {rmse_true:<10.4f} {r2_true:<10.4f}")
    
    return X, y, selected_indices, lasso_selected

# Demonstrate feature selection using QR decomposition
X_fs, y_fs, qr_selected, lasso_selected = demonstrate_feature_selection()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored various applications of QR decomposition in computational linear algebra and data science:
# 
# 1. **Solving Linear Systems**: We demonstrated how QR decomposition can be used to solve systems of linear equations, comparing its performance with other methods.
# 
# 2. **Least Squares Problems**: We showed how QR decomposition provides a numerically stable way to solve least squares problems without explicitly forming the normal equations.
# 
# 3. **QR Algorithm for Eigenvalues**: We implemented and visualized the QR algorithm for computing eigenvalues, including an enhanced version with shifts for faster convergence.
# 
# 4. **Singular Value Decomposition (SVD)**: We demonstrated how QR decomposition can be used as a building block for computing the SVD of a matrix.
# 
# 5. **Data Science Applications**: We explored how QR decomposition can be used for dimensionality reduction (PCA) and feature selection.
# 
# Key advantages of QR decomposition in these applications:
# 
# - **Numerical stability**: QR decomposition preserves orthogonality, making it numerically stable for a wide range of problems.
# - **Versatility**: It serves as a building block for many other important algorithms in numerical linear algebra.
# - **Efficiency**: For certain problems, QR-based methods can be more computationally efficient than alternatives.
# 
# The main limitation is the computational cost (O(mn²) for an m×n matrix), which can be prohibitive for very large matrices. However, for many practical applications, the numerical stability advantages of QR decomposition outweigh the computational cost.

# %%