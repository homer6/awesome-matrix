# %% [markdown]
# # Singular Value Decomposition: Algorithms
# 
# This notebook explores different algorithms for computing the Singular Value Decomposition (SVD) and their implementation details. We'll examine the tradeoffs between these methods in terms of computational efficiency, numerical stability, and accuracy.
# 
# We'll focus on:
# 
# 1. **Power Method for SVD**
# 2. **Bidiagonalization Method**
# 3. **Jacobi SVD Algorithm**
# 4. **Randomized SVD**
# 5. **Block and Parallelized SVD**
# 
# Each algorithm has specific strengths and weaknesses that make it suitable for different types of matrices and applications.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg
import time
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

# Helper functions for visualization
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

# Functions to create example matrices
def create_example_matrix(m=4, n=3, method="random"):
    """Create a matrix for SVD demonstration."""
    if method == "random":
        # Create a random matrix
        A = torch.rand(m, n) * 10 - 5  # Values between -5 and 5
    elif method == "simple":
        # Simple predefined matrix for clear demonstration
        if m == 3 and n == 2:
            A = torch.tensor([
                [4.0, 0.0],
                [3.0, -5.0],
                [0.0, 4.0]
            ])
        else:
            raise ValueError(f"Simple method doesn't support dimensions {m}x{n}")
    elif method == "image":
        # Create a simple image-like matrix with structure
        A = torch.zeros(m, n)
        # Add a gradient pattern
        for i in range(m):
            for j in range(n):
                A[i, j] = i + j * 0.5
        # Add some noise
        A = A + torch.randn(m, n) * 0.2
    elif method == "lowrank":
        # Create a low-rank matrix
        rank = min(min(m, n) - 1, 5)  # low rank, but not too low
        # Create factors
        U = torch.randn(m, rank)
        V = torch.randn(n, rank)
        # Create matrix as a sum of rank-1 matrices
        A = U @ V.T
    else:
        raise ValueError("Unknown method")
    
    return A

# Function to evaluate SVD quality
def evaluate_svd(A, U, S, V):
    """
    Evaluate the quality of an SVD decomposition.
    
    Args:
        A: Original matrix
        U: Left singular vectors
        S: Singular values (or diagonal matrix)
        V: Right singular vectors
        
    Returns:
        Dict with reconstruction error and orthogonality measures
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    # Handle different forms of S (vector or matrix)
    if len(S.shape) == 1:
        # S is a vector, create a diagonal matrix
        m, n = A.shape
        S_matrix = torch.zeros(m, n, dtype=A.dtype)
        for i in range(min(m, n)):
            if i < len(S):
                S_matrix[i, i] = S[i]
    else:
        # S is already a matrix
        S_matrix = S
    
    # Handle different forms of V (V or V^T)
    if V.shape[0] == A.shape[1]:
        # V is provided (columns are eigenvectors)
        V_matrix = V
    else:
        # V^T is provided
        V_matrix = V.T
    
    # Reconstruction error
    A_reconstructed = U @ S_matrix @ V_matrix.T
    reconstruction_error = torch.norm(A - A_reconstructed).item() / torch.norm(A).item()
    
    # Orthogonality of U
    U_orthogonality_error = torch.norm(U.T @ U - torch.eye(U.shape[1], dtype=U.dtype)).item()
    
    # Orthogonality of V
    V_orthogonality_error = torch.norm(V_matrix.T @ V_matrix - torch.eye(V_matrix.shape[1], dtype=V_matrix.dtype)).item()
    
    return {
        'reconstruction_error': reconstruction_error,
        'U_orthogonality_error': U_orthogonality_error,
        'V_orthogonality_error': V_orthogonality_error
    }

# %% [markdown]
# ## 1. Power Method for SVD
# 
# The power method is a simple iterative approach for finding the largest singular value and its corresponding singular vectors. The basic idea is:
# 
# 1. Start with a random vector $v$
# 2. Repeatedly apply $A^TA$ to $v$ (i.e., compute $(A^TA)^k v$ for increasing $k$)
# 3. The vector will converge to the eigenvector corresponding to the largest eigenvalue of $A^TA$
# 4. This gives the right singular vector corresponding to the largest singular value
# 
# By deflation (subtracting the component along the found singular vectors), we can find subsequent singular values and vectors.
# 
# Let's implement the power method for SVD:

# %%
def power_method_svd(A, k=None, max_iter=100, tol=1e-8):
    """
    Compute SVD using the power method.
    
    Args:
        A: Input matrix
        k: Number of singular values/vectors to compute (default: min(m,n))
        max_iter: Maximum number of iterations for each singular value
        tol: Convergence tolerance
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    if k is None:
        k = min(m, n)
    
    # Initialize output matrices
    U = torch.zeros((m, k), dtype=A.dtype)
    S = torch.zeros(k, dtype=A.dtype)
    V = torch.zeros((n, k), dtype=A.dtype)
    
    # Make a copy of A for deflation
    A_deflated = A.clone()
    
    # Compute SVD one component at a time
    for i in range(k):
        # Initialize a random vector
        v = torch.randn(n, dtype=A.dtype)
        v = v / torch.norm(v)
        
        # Apply power iteration
        for _ in range(max_iter):
            # v = (A^T A) v
            v_new = A_deflated.T @ (A_deflated @ v)
            
            # Normalize
            v_new_norm = torch.norm(v_new)
            if v_new_norm < tol:
                # If we get a zero vector, the remaining singular values are zero
                break
                
            v_new = v_new / v_new_norm
            
            # Check for convergence
            if torch.norm(v_new - v) < tol or torch.norm(v_new + v) < tol:
                break
                
            v = v_new
        
        # Compute singular value and left singular vector
        u = A_deflated @ v
        s = torch.norm(u)
        
        if s > tol:
            u = u / s
            
            # Store results
            U[:, i] = u
            S[i] = s
            V[:, i] = v
            
            # Deflate the matrix: A_next = A - s*u*v^T
            A_deflated = A_deflated - s * torch.outer(u, v)
        else:
            # Zero (or very small) singular value
            break
    
    # Return only the non-zero components
    non_zero = torch.sum(S > tol).item()
    return U[:, :non_zero], S[:non_zero], V[:, :non_zero]

def demonstrate_power_method():
    """Demonstrate the power method for SVD."""
    # Create a simple matrix
    A = torch.tensor([
        [3.0, 2.0, 2.0],
        [2.0, 3.0, -2.0]
    ])
    
    # Display the original matrix
    plot_matrix(A, "Original Matrix")
    
    # Compute SVD using power method
    U_power, S_power, V_power = power_method_svd(A)
    
    # For comparison, use torch.linalg.svd
    U_torch, S_torch, V_torch = torch.linalg.svd(A, full_matrices=False)
    
    # Display the results
    print("Power Method SVD:")
    print("Left Singular Vectors (U):")
    print(U_power.numpy())
    print("\nSingular Values (S):")
    print(S_power.numpy())
    print("\nRight Singular Vectors (V):")
    print(V_power.numpy())
    
    print("\nPyTorch SVD:")
    print("Left Singular Vectors (U):")
    print(U_torch.numpy())
    print("\nSingular Values (S):")
    print(S_torch.numpy())
    print("\nRight Singular Vectors (V):")
    print(V_torch.T.numpy())
    
    # Create matrices for visualization
    m, n = A.shape
    S_power_matrix = torch.zeros(m, n)
    for i in range(len(S_power)):
        S_power_matrix[i, i] = S_power[i]
    
    # Visualize the power method SVD
    plot_matrix(U_power, "Left Singular Vectors (U) - Power Method")
    plot_matrix(S_power_matrix, "Singular Values (Σ) - Power Method")
    plot_matrix(V_power, "Right Singular Vectors (V) - Power Method")
    
    # Reconstruct the matrix
    A_reconstructed = U_power @ S_power_matrix @ V_power.T
    plot_matrix(A_reconstructed, "Reconstructed Matrix - Power Method")
    
    # Calculate errors
    power_evaluation = evaluate_svd(A, U_power, S_power, V_power)
    torch_evaluation = evaluate_svd(A, U_torch, S_torch, V_torch)
    
    print("\nReconstruction Error:")
    print(f"Power Method: {power_evaluation['reconstruction_error']:.2e}")
    print(f"PyTorch SVD: {torch_evaluation['reconstruction_error']:.2e}")
    
    print("\nU Orthogonality Error:")
    print(f"Power Method: {power_evaluation['U_orthogonality_error']:.2e}")
    print(f"PyTorch SVD: {torch_evaluation['U_orthogonality_error']:.2e}")
    
    print("\nV Orthogonality Error:")
    print(f"Power Method: {power_evaluation['V_orthogonality_error']:.2e}")
    print(f"PyTorch SVD: {torch_evaluation['V_orthogonality_error']:.2e}")
    
    # Visualize the convergence of power method
    def visualize_power_convergence(A, max_iter=20):
        """Visualize how the power method converges to singular values."""
        m, n = A.shape
        
        # Initialize a random vector
        v = torch.randn(n, dtype=A.dtype)
        v = v / torch.norm(v)
        
        # Keep track of approximations at each iteration
        approximations = []
        
        # Apply power iteration
        for i in range(max_iter):
            # v = (A^T A) v
            v_new = A.T @ (A @ v)
            
            # Compute the Rayleigh quotient (approximation of largest eigenvalue)
            rayleigh = (v_new @ v) / (v @ v)
            
            # Normalize
            v_new = v_new / torch.norm(v_new)
            
            # Compute current approximation of largest singular value
            u = A @ v
            sigma = torch.norm(u)
            u = u / sigma
            
            # Store approximation
            approximations.append(sigma.item())
            
            v = v_new
        
        # Get the true largest singular value
        _, S_true, _ = torch.linalg.svd(A)
        largest_sv = S_true[0].item()
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_iter+1), approximations, 'o-', label='Power Method Approximation')
        plt.axhline(y=largest_sv, color='r', linestyle='--', label=f'True Value ({largest_sv:.4f})')
        plt.xlabel("Iteration")
        plt.ylabel("Approximation of Largest Singular Value")
        plt.title("Convergence of Power Method")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        # Plot relative error
        plt.figure(figsize=(10, 6))
        relative_errors = [abs(approx - largest_sv) / largest_sv for approx in approximations]
        plt.semilogy(range(1, max_iter+1), relative_errors, 'o-')
        plt.xlabel("Iteration")
        plt.ylabel("Relative Error")
        plt.title("Convergence Rate of Power Method")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return approximations
    
    # Visualize convergence for our example matrix
    approximations = visualize_power_convergence(A)
    
    return A, U_power, S_power, V_power, approximations

# Demonstrate power method for SVD
A_power, U_power, S_power, V_power, approximations_power = demonstrate_power_method()

# %% [markdown]
# ### 1.1 Shifted Power Method for Convergence Acceleration
# 
# The basic power method can be slow to converge, especially when the largest singular values are close in magnitude. The shifted power method can accelerate convergence by applying a shift:

# %%
def shifted_power_method_svd(A, k=None, max_iter=100, tol=1e-8):
    """
    Compute SVD using the shifted power method for faster convergence.
    
    Args:
        A: Input matrix
        k: Number of singular values/vectors to compute (default: min(m,n))
        max_iter: Maximum number of iterations for each singular value
        tol: Convergence tolerance
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    if k is None:
        k = min(m, n)
    
    # Initialize output matrices
    U = torch.zeros((m, k), dtype=A.dtype)
    S = torch.zeros(k, dtype=A.dtype)
    V = torch.zeros((n, k), dtype=A.dtype)
    
    # Make a copy of A for deflation
    A_deflated = A.clone()
    
    # Compute SVD one component at a time
    for i in range(k):
        # Estimate the largest singular value using a few iterations of the power method
        v = torch.randn(n, dtype=A.dtype)
        v = v / torch.norm(v)
        
        for _ in range(5):  # Just a few iterations to get an estimate
            v = A_deflated.T @ (A_deflated @ v)
            v = v / torch.norm(v)
        
        # Estimate the largest eigenvalue of A^T A
        rayleigh = v @ (A_deflated.T @ (A_deflated @ v)) / (v @ v)
        
        # Use a shift slightly smaller than the estimated largest eigenvalue
        shift = 0.95 * rayleigh
        
        # Initialize a random vector
        v = torch.randn(n, dtype=A.dtype)
        v = v / torch.norm(v)
        
        # Apply shifted power iteration
        for _ in range(max_iter):
            # v = (A^T A - shift*I)^-1 v
            # This is equivalent to solving (A^T A - shift*I) x = v
            # But for simplicity, we'll just use the shifted iteration without inversion
            
            v_new = A_deflated.T @ (A_deflated @ v) - shift * v
            
            # Normalize
            v_new_norm = torch.norm(v_new)
            if v_new_norm < tol:
                break
                
            v_new = v_new / v_new_norm
            
            # Check for convergence
            if torch.norm(v_new - v) < tol or torch.norm(v_new + v) < tol:
                break
                
            v = v_new
        
        # Compute singular value and left singular vector
        u = A_deflated @ v
        s = torch.norm(u)
        
        if s > tol:
            u = u / s
            
            # Store results
            U[:, i] = u
            S[i] = s
            V[:, i] = v
            
            # Deflate the matrix: A_next = A - s*u*v^T
            A_deflated = A_deflated - s * torch.outer(u, v)
        else:
            # Zero (or very small) singular value
            break
    
    # Return only the non-zero components
    non_zero = torch.sum(S > tol).item()
    return U[:, :non_zero], S[:non_zero], V[:, :non_zero]

def compare_power_methods():
    """Compare regular and shifted power methods."""
    # Create a matrix with close singular values
    A = torch.tensor([
        [3.0, 2.0, 1.0],
        [2.0, 3.0, 1.0],
        [1.0, 1.0, 3.0]
    ])
    
    # Compute true singular values
    _, S_true, _ = torch.linalg.svd(A)
    
    print("True Singular Values:")
    print(S_true.numpy())
    
    # Compare convergence rates for largest singular value
    def compare_convergence(A, max_iter=30):
        """Compare convergence rates of regular and shifted power methods."""
        m, n = A.shape
        
        # Regular power method
        v_regular = torch.randn(n, dtype=A.dtype)
        v_regular = v_regular / torch.norm(v_regular)
        regular_approximations = []
        
        for i in range(max_iter):
            v_regular = A.T @ (A @ v_regular)
            v_regular = v_regular / torch.norm(v_regular)
            u_regular = A @ v_regular
            sigma_regular = torch.norm(u_regular)
            regular_approximations.append(sigma_regular.item())
        
        # Shifted power method
        v_shifted = torch.randn(n, dtype=A.dtype)
        v_shifted = v_shifted / torch.norm(v_shifted)
        shifted_approximations = []
        
        # Estimate the largest eigenvalue
        v_est = v_shifted.clone()
        for _ in range(5):
            v_est = A.T @ (A @ v_est)
            v_est = v_est / torch.norm(v_est)
        
        rayleigh = v_est @ (A.T @ (A @ v_est)) / (v_est @ v_est)
        shift = 0.95 * rayleigh
        
        for i in range(max_iter):
            v_shifted = A.T @ (A @ v_shifted) - shift * v_shifted
            v_shifted = v_shifted / torch.norm(v_shifted)
            u_shifted = A @ v_shifted
            sigma_shifted = torch.norm(u_shifted)
            shifted_approximations.append(sigma_shifted.item())
        
        # Plot convergence comparison
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, max_iter+1), regular_approximations, 'o-', label='Regular Power Method')
        plt.plot(range(1, max_iter+1), shifted_approximations, 's-', label='Shifted Power Method')
        plt.axhline(y=S_true[0].item(), color='r', linestyle='--', label=f'True Value ({S_true[0].item():.4f})')
        plt.xlabel("Iteration")
        plt.ylabel("Approximation of Largest Singular Value")
        plt.title("Convergence Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        regular_errors = [abs(approx - S_true[0].item()) / S_true[0].item() for approx in regular_approximations]
        shifted_errors = [abs(approx - S_true[0].item()) / S_true[0].item() for approx in shifted_approximations]
        
        plt.semilogy(range(1, max_iter+1), regular_errors, 'o-', label='Regular Power Method')
        plt.semilogy(range(1, max_iter+1), shifted_errors, 's-', label='Shifted Power Method')
        plt.xlabel("Iteration")
        plt.ylabel("Relative Error (log scale)")
        plt.title("Convergence Rate Comparison")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Return the number of iterations needed to reach a given accuracy
        target_error = 1e-6
        regular_iter = next((i for i, error in enumerate(regular_errors) if error < target_error), max_iter)
        shifted_iter = next((i for i, error in enumerate(shifted_errors) if error < target_error), max_iter)
        
        print(f"Iterations to reach {target_error:.1e} relative error:")
        print(f"Regular Power Method: {regular_iter + 1}")
        print(f"Shifted Power Method: {shifted_iter + 1}")
        
        return regular_approximations, shifted_approximations
    
    # Compare convergence rates
    regular_approx, shifted_approx = compare_convergence(A)
    
    # Compare full SVD computation timing
    def time_methods(A, k=None):
        """Compare timing of regular and shifted power methods."""
        # Time regular power method
        start_time = time.time()
        U_regular, S_regular, V_regular = power_method_svd(A, k=k)
        regular_time = time.time() - start_time
        
        # Time shifted power method
        start_time = time.time()
        U_shifted, S_shifted, V_shifted = shifted_power_method_svd(A, k=k)
        shifted_time = time.time() - start_time
        
        # Time PyTorch SVD
        start_time = time.time()
        U_torch, S_torch, V_torch = torch.linalg.svd(A)
        torch_time = time.time() - start_time
        
        # Calculate errors
        regular_evaluation = evaluate_svd(A, U_regular, S_regular, V_regular)
        shifted_evaluation = evaluate_svd(A, U_shifted, S_shifted, V_shifted)
        torch_evaluation = evaluate_svd(A, U_torch, S_torch, V_torch)
        
        print("\nTiming Comparison:")
        print(f"Regular Power Method: {regular_time:.6f} seconds")
        print(f"Shifted Power Method: {shifted_time:.6f} seconds")
        print(f"PyTorch SVD: {torch_time:.6f} seconds")
        
        print("\nReconstruction Error:")
        print(f"Regular Power Method: {regular_evaluation['reconstruction_error']:.2e}")
        print(f"Shifted Power Method: {shifted_evaluation['reconstruction_error']:.2e}")
        print(f"PyTorch SVD: {torch_evaluation['reconstruction_error']:.2e}")
        
        # Plot results
        methods = ['Regular Power', 'Shifted Power', 'PyTorch SVD']
        times = [regular_time, shifted_time, torch_time]
        errors = [regular_evaluation['reconstruction_error'], 
                  shifted_evaluation['reconstruction_error'],
                  torch_evaluation['reconstruction_error']]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(methods, times)
        plt.ylabel("Time (seconds)")
        plt.title("Computation Time")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(methods, errors)
        plt.ylabel("Reconstruction Error")
        plt.title("Accuracy")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'methods': methods,
            'times': times,
            'errors': errors,
            'results': {
                'regular': (U_regular, S_regular, V_regular),
                'shifted': (U_shifted, S_shifted, V_shifted),
                'torch': (U_torch, S_torch, V_torch)
            }
        }
    
    # Compare timing and accuracy
    comparison_results = time_methods(A)
    
    return A, comparison_results

# Compare regular and shifted power methods
A_shifted, comparison_shifted = compare_power_methods()

# %% [markdown]
# ## 2. Bidiagonalization Method
# 
# The bidiagonalization method, also known as Golub-Kahan-Lanczos bidiagonalization, is a more efficient approach for computing SVD, especially for large sparse matrices. It first reduces the matrix to a bidiagonal form, and then computes the SVD of the bidiagonal matrix.
# 
# The basic algorithm has two phases:
# 1. Reduce $A$ to a bidiagonal form $B$ using orthogonal transformations: $A = UBV^T$
# 2. Compute the SVD of $B$: $B = \hat{U} \Sigma \hat{V}^T$
# 3. Combine the transformations: $A = (U\hat{U}) \Sigma (\hat{V}^T V^T)$

# %%
def bidiagonalization(A, full_matrices=True):
    """
    Reduce a matrix to bidiagonal form using Householder reflections.
    
    Args:
        A: Input matrix
        full_matrices: Whether to return full or economy-sized U and V
        
    Returns:
        U: Left orthogonal matrix
        B: Bidiagonal matrix
        V: Right orthogonal matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    k = min(m, n)
    
    # Initialize U, B, and V
    if full_matrices:
        U = torch.eye(m, dtype=A.dtype)
        V = torch.eye(n, dtype=A.dtype)
    else:
        U = torch.eye(m, k, dtype=A.dtype)
        V = torch.eye(n, k, dtype=A.dtype)
    
    # Create a copy of A to work with
    B = A.clone()
    
    # Bidiagonalization process
    for i in range(k):
        # Apply Householder reflection to zero elements below the diagonal
        if i < m-1:
            # Extract the column vector
            x = B[i:, i]
            
            # Construct the Householder vector
            alpha = torch.norm(x)
            if x[0] < 0:
                alpha = -alpha
            
            # Handle the case where x is already a multiple of the first unit vector
            if alpha == 0 or (x.shape[0] == 1 and x[0] == alpha):
                continue
                
            u = x.clone()
            u[0] = u[0] + alpha
            u = u / torch.norm(u)
            
            # Apply the Householder reflection to B
            B[i:, i:] = B[i:, i:] - 2.0 * torch.outer(u, (u @ B[i:, i:]))
            
            # Update U
            if full_matrices:
                U[:, i:] = U[:, i:] - 2.0 * torch.outer(U[:, i:] @ u, u)
            else:
                U[:, i:k] = U[:, i:k] - 2.0 * torch.outer(U[:, i:k] @ u, u[:k-i])
        
        # Apply Householder reflection to zero elements to the right of the superdiagonal
        if i < k-1 and i < n-1:
            # Extract the row vector
            x = B[i, i+1:]
            
            # Construct the Householder vector
            alpha = torch.norm(x)
            if x[0] < 0:
                alpha = -alpha
            
            # Handle the case where x is already a multiple of the first unit vector
            if alpha == 0 or (x.shape[0] == 1 and x[0] == alpha):
                continue
                
            u = x.clone()
            u[0] = u[0] + alpha
            u = u / torch.norm(u)
            
            # Apply the Householder reflection to B
            B[i:, i+1:] = B[i:, i+1:] - 2.0 * torch.outer((B[i:, i+1:] @ u), u)
            
            # Update V
            if full_matrices:
                V[:, i+1:] = V[:, i+1:] - 2.0 * torch.outer(V[:, i+1:] @ u, u)
            else:
                V[:, (i+1):k] = V[:, (i+1):k] - 2.0 * torch.outer(V[:, (i+1):k] @ u, u[:k-(i+1)])
    
    return U, B, V

def svd_of_bidiagonal(B, tol=1e-8, max_iter=100):
    """
    Compute the SVD of a bidiagonal matrix using the QR algorithm.
    
    Args:
        B: Bidiagonal matrix
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(B, np.ndarray):
        B = torch.tensor(B, dtype=torch.float64)
    
    m, n = B.shape
    k = min(m, n)
    
    # Initialize U and V as identity matrices
    U = torch.eye(m, dtype=B.dtype)
    V = torch.eye(n, dtype=B.dtype)
    
    # Make a copy of B to work with
    B_work = B.clone()
    
    # Apply QR algorithm to B^T B to find its eigenvalues and eigenvectors
    # This is a simplified approach; in practice, specialized algorithms for
    # bidiagonal SVD would be used (like the implicit QR algorithm with shifts)
    
    # For simplicity, we'll use a direct eigendecomposition here
    BTB = B_work.T @ B_work
    eigvals, eigvecs = torch.linalg.eigh(BTB)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    
    # Singular values are square roots of eigenvalues of B^T B
    S = torch.sqrt(torch.clamp(eigvals, min=0))
    
    # Right singular vectors are eigenvectors of B^T B
    V = eigvecs
    
    # Compute left singular vectors: u_i = (1/sigma_i) * B * v_i
    U = torch.zeros((m, k), dtype=B.dtype)
    for i in range(k):
        if S[i] > tol:
            U[:, i] = (B_work @ V[:, i]) / S[i]
        else:
            # For zero singular values, use an arbitrary orthogonal vector
            if i == 0:
                U[:, i] = torch.zeros(m, dtype=B.dtype)
                U[0, i] = 1.0
            else:
                # Make orthogonal to the previous vectors
                u = torch.randn(m, dtype=B.dtype)
                for j in range(i):
                    u = u - torch.dot(u, U[:, j]) * U[:, j]
                U[:, i] = u / torch.norm(u)
    
    return U, S, V

def bidiagonal_svd(A, full_matrices=True):
    """
    Compute SVD using bidiagonalization followed by SVD of the bidiagonal matrix.
    
    Args:
        A: Input matrix
        full_matrices: Whether to return full or economy-sized U and V
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    # Step 1: Reduce A to bidiagonal form
    U_bidiag, B, V_bidiag = bidiagonalization(A, full_matrices=full_matrices)
    
    # Step 2: Compute SVD of the bidiagonal matrix
    U_svd, S, V_svd = svd_of_bidiagonal(B)
    
    # Step 3: Combine the transformations
    U = U_bidiag @ U_svd
    V = V_bidiag @ V_svd
    
    return U, S, V

def demonstrate_bidiagonalization():
    """Demonstrate the bidiagonalization method for SVD."""
    # Create a small matrix
    A = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ])
    
    # Display the original matrix
    plot_matrix(A, "Original Matrix")
    
    # Step 1: Reduce to bidiagonal form
    U_bidiag, B, V_bidiag = bidiagonalization(A)
    
    # Display the bidiagonal matrix
    plot_matrix(B, "Bidiagonal Matrix B")
    
    # Verify that A = U_bidiag @ B @ V_bidiag.T
    A_reconstructed = U_bidiag @ B @ V_bidiag.T
    plot_matrix(A_reconstructed, "Reconstructed Matrix (U_bidiag @ B @ V_bidiag^T)")
    
    # Check reconstruction error
    recon_error = torch.norm(A - A_reconstructed).item() / torch.norm(A).item()
    print(f"Bidiagonalization reconstruction error: {recon_error:.2e}")
    
    # Step 2: Compute SVD of the bidiagonal matrix
    U_svd, S, V_svd = svd_of_bidiagonal(B)
    
    # Combine the transformations
    U = U_bidiag @ U_svd
    V = V_bidiag @ V_svd
    
    # Create a diagonal matrix from S
    m, n = A.shape
    S_matrix = torch.zeros(m, n)
    for i in range(min(m, n)):
        if i < len(S):
            S_matrix[i, i] = S[i]
    
    # Display the SVD components
    plot_matrix(U, "Left Singular Vectors (U)")
    plot_matrix(S_matrix, "Singular Values (Σ)")
    plot_matrix(V, "Right Singular Vectors (V)")
    
    # Reconstruct the matrix from SVD
    A_svd = U @ S_matrix @ V.T
    plot_matrix(A_svd, "Reconstructed Matrix (U @ Σ @ V^T)")
    
    # Check reconstruction error
    svd_error = torch.norm(A - A_svd).item() / torch.norm(A).item()
    print(f"SVD reconstruction error: {svd_error:.2e}")
    
    # For comparison, use torch.linalg.svd
    U_torch, S_torch, V_torch = torch.linalg.svd(A, full_matrices=True)
    
    # Create a diagonal matrix from S_torch
    S_torch_matrix = torch.zeros(m, n)
    for i in range(min(m, n)):
        S_torch_matrix[i, i] = S_torch[i]
    
    # Reconstruct using PyTorch SVD
    A_torch = U_torch @ S_torch_matrix @ V_torch
    torch_error = torch.norm(A - A_torch).item() / torch.norm(A).item()
    
    print(f"PyTorch SVD reconstruction error: {torch_error:.2e}")
    
    # Compare singular values
    print("\nSingular Values:")
    print("Bidiagonal SVD:", S.numpy())
    print("PyTorch SVD:", S_torch.numpy())
    
    return A, U, S, V, B

# Demonstrate bidiagonalization method
A_bidiag, U_bidiag, S_bidiag, V_bidiag, B_bidiag = demonstrate_bidiagonalization()

# %% [markdown]
# ## 3. Jacobi SVD Algorithm
# 
# The Jacobi SVD algorithm is an iterative method that directly computes the SVD by applying a sequence of plane rotations (Givens rotations) to diagonalize the matrix. Unlike the previous methods, it doesn't reduce the matrix to a bidiagonal form first.
# 
# The algorithm focuses on eliminating off-diagonal elements of $A^TA$ by iteratively applying rotations to both rows and columns:

# %%
def jacobi_svd(A, tol=1e-8, max_iter=100):
    """
    Compute SVD using the Jacobi method.
    
    Args:
        A: Input matrix
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    # Initialize U and V as identity matrices
    U = torch.eye(m, dtype=A.dtype)
    V = torch.eye(n, dtype=A.dtype)
    
    # Make a copy of A to work with
    B = A.clone()
    
    # Compute the squared Frobenius norm of the off-diagonal elements of B^T B
    BtB = B.T @ B
    off_diag_norm = torch.sum(BtB**2) - torch.sum(torch.diag(BtB)**2)
    
    # Iterate until convergence
    num_iter = 0
    while off_diag_norm > tol and num_iter < max_iter:
        # Find the largest off-diagonal element of B^T B
        p, q = 0, 1  # Default indices
        max_val = 0.0
        
        for i in range(n):
            for j in range(i+1, n):
                if abs(BtB[i, j]) > max_val:
                    max_val = abs(BtB[i, j])
                    p, q = i, j
        
        # Compute the Jacobi rotation parameters
        if max_val < tol:
            break
        
        # Compute the Jacobi rotation
        alpha = BtB[p, p]
        beta = BtB[q, q]
        gamma = BtB[p, q]
        
        if abs(gamma) < tol:
            continue
        
        # Compute the cosine and sine of the rotation angle
        if abs(alpha - beta) < tol:
            c = 1.0 / np.sqrt(2)
            s = c
        else:
            zeta = (beta - alpha) / (2 * gamma)
            t = 1.0 / (abs(zeta) + np.sqrt(1 + zeta**2))
            if zeta < 0:
                t = -t
            c = 1.0 / np.sqrt(1 + t**2)
            s = t * c
        
        # Create the Givens rotation matrix
        G = torch.eye(n, dtype=A.dtype)
        G[p, p] = c
        G[p, q] = -s
        G[q, p] = s
        G[q, q] = c
        
        # Apply the rotation to B and update V
        B = B @ G
        V = V @ G
        
        # Recompute B^T B and the off-diagonal norm
        BtB = B.T @ B
        off_diag_norm = torch.sum(BtB**2) - torch.sum(torch.diag(BtB)**2)
        
        num_iter += 1
    
    # Compute the singular values and left singular vectors
    S = torch.zeros(min(m, n), dtype=A.dtype)
    for i in range(min(m, n)):
        S[i] = torch.norm(B[:, i])
        if S[i] > tol:
            U[:, i] = B[:, i] / S[i]
        else:
            # For zero singular values, use an arbitrary orthogonal vector
            if i == 0:
                U[:, i] = torch.zeros(m, dtype=A.dtype)
                U[0, i] = 1.0
            else:
                # Make orthogonal to the previous vectors
                u = torch.randn(m, dtype=A.dtype)
                for j in range(i):
                    u = u - torch.dot(u, U[:, j]) * U[:, j]
                U[:, i] = u / torch.norm(u)
    
    # Sort singular values and vectors in descending order
    sorted_indices = torch.argsort(S, descending=True)
    S = S[sorted_indices]
    U = U[:, sorted_indices]
    V = V[:, sorted_indices]
    
    return U, S, V

def demonstrate_jacobi_svd():
    """Demonstrate the Jacobi SVD algorithm."""
    # Create a small matrix
    A = torch.tensor([
        [3.0, 2.0, 2.0],
        [2.0, 3.0, -2.0]
    ])
    
    # Display the original matrix
    plot_matrix(A, "Original Matrix")
    
    # Compute SVD using Jacobi method
    U_jacobi, S_jacobi, V_jacobi = jacobi_svd(A)
    
    # Create a diagonal matrix from S
    m, n = A.shape
    S_matrix = torch.zeros(m, n)
    for i in range(min(m, n)):
        S_matrix[i, i] = S_jacobi[i]
    
    # Display the SVD components
    plot_matrix(U_jacobi, "Left Singular Vectors (U) - Jacobi")
    plot_matrix(S_matrix, "Singular Values (Σ) - Jacobi")
    plot_matrix(V_jacobi, "Right Singular Vectors (V) - Jacobi")
    
    # Reconstruct the matrix from SVD
    A_jacobi = U_jacobi @ S_matrix @ V_jacobi.T
    plot_matrix(A_jacobi, "Reconstructed Matrix (U @ Σ @ V^T) - Jacobi")
    
    # Check reconstruction error
    jacobi_error = torch.norm(A - A_jacobi).item() / torch.norm(A).item()
    print(f"Jacobi SVD reconstruction error: {jacobi_error:.2e}")
    
    # For comparison, use torch.linalg.svd
    U_torch, S_torch, V_torch = torch.linalg.svd(A, full_matrices=True)
    
    # Create a diagonal matrix from S_torch
    S_torch_matrix = torch.zeros(m, n)
    for i in range(min(m, n)):
        S_torch_matrix[i, i] = S_torch[i]
    
    # Reconstruct using PyTorch SVD
    A_torch = U_torch @ S_torch_matrix @ V_torch
    torch_error = torch.norm(A - A_torch).item() / torch.norm(A).item()
    
    print(f"PyTorch SVD reconstruction error: {torch_error:.2e}")
    
    # Compare singular values
    print("\nSingular Values:")
    print("Jacobi SVD:", S_jacobi.numpy())
    print("PyTorch SVD:", S_torch.numpy())
    
    # Compare orthogonality
    jacobi_eval = evaluate_svd(A, U_jacobi, S_jacobi, V_jacobi)
    torch_eval = evaluate_svd(A, U_torch, S_torch, V_torch)
    
    print("\nU Orthogonality Error:")
    print(f"Jacobi SVD: {jacobi_eval['U_orthogonality_error']:.2e}")
    print(f"PyTorch SVD: {torch_eval['U_orthogonality_error']:.2e}")
    
    print("\nV Orthogonality Error:")
    print(f"Jacobi SVD: {jacobi_eval['V_orthogonality_error']:.2e}")
    print(f"PyTorch SVD: {torch_eval['V_orthogonality_error']:.2e}")
    
    return A, U_jacobi, S_jacobi, V_jacobi

# Demonstrate Jacobi SVD
A_jacobi, U_jacobi, S_jacobi, V_jacobi = demonstrate_jacobi_svd()

# %% [markdown]
# ## 4. Randomized SVD
# 
# Randomized SVD is a modern approach that uses random projections to efficiently compute an approximate SVD, particularly for large matrices. The idea is to first identify a low-dimensional subspace that captures most of the action of the matrix, and then compute the SVD within this subspace.
# 
# The basic algorithm is:
# 1. Generate a random matrix $\Omega$ with $r$ columns
# 2. Form $Y = A\Omega$ to sample the range of $A$
# 3. Orthogonalize $Y$ to get a basis $Q$ for the range of $A$
# 4. Form $B = Q^T A$, which is a smaller matrix
# 5. Compute the SVD of $B = \tilde{U} \Sigma V^T$
# 6. Compute $U = Q \tilde{U}$

# %%
def randomized_svd(A, k, p=5, q=2):
    """
    Compute randomized SVD.
    
    Args:
        A: Input matrix
        k: Target rank
        p: Oversampling parameter
        q: Number of power iterations
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    # Step 1: Generate a random matrix
    r = min(k + p, min(m, n))  # Rank with oversampling
    Omega = torch.randn(n, r, dtype=A.dtype)
    
    # Step 2: Sample the range of A
    Y = A @ Omega
    
    # Optional: Power iterations to improve accuracy for low-rank approximation
    for _ in range(q):
        Y = A @ (A.T @ Y)
    
    # Step 3: Orthogonalize Y to get a basis Q
    Q, _ = torch.linalg.qr(Y, mode='reduced')
    
    # Step 4: Form the smaller matrix B
    B = Q.T @ A
    
    # Step 5: Compute SVD of B
    Uhat, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Step 6: Compute U
    U = Q @ Uhat
    
    # Keep only the top k components
    return U[:, :k], S[:k], Vt[:k, :]

def demonstrate_randomized_svd():
    """Demonstrate randomized SVD algorithm."""
    # Create a low-rank matrix (rank 3 in a 50x30 matrix)
    m, n = 50, 30
    rank = 3
    
    # Create factors
    U_true = torch.nn.functional.normalize(torch.randn(m, rank), dim=0)
    V_true = torch.nn.functional.normalize(torch.randn(n, rank), dim=0)
    S_true = torch.tensor([10.0, 5.0, 1.0])  # Singular values
    
    # Create the matrix
    A = U_true @ torch.diag(S_true) @ V_true.T
    
    # Add some noise
    noise_level = 0.01
    A_noisy = A + noise_level * torch.randn(m, n)
    
    # Compute full SVD using PyTorch
    start_time = time.time()
    U_full, S_full, V_full = torch.linalg.svd(A_noisy, full_matrices=False)
    full_time = time.time() - start_time
    
    # Compute randomized SVD
    start_time = time.time()
    U_rand, S_rand, V_rand = randomized_svd(A_noisy, k=rank, p=5, q=2)
    rand_time = time.time() - start_time
    
    # Compare timing and accuracy
    print("Matrix size:", A.shape)
    print(f"Full SVD time: {full_time:.6f} seconds")
    print(f"Randomized SVD time: {rand_time:.6f} seconds")
    print(f"Speedup: {full_time / rand_time:.2f}x")
    
    # Convert V_rand from V^T to V
    V_rand = V_rand.T
    
    # Evaluate accuracy
    full_eval = evaluate_svd(A_noisy, U_full[:, :rank], S_full[:rank], V_full[:, :rank])
    rand_eval = evaluate_svd(A_noisy, U_rand, S_rand, V_rand)
    
    print("\nReconstruction Error (with noise):")
    print(f"Full SVD (top {rank} components): {full_eval['reconstruction_error']:.2e}")
    print(f"Randomized SVD (rank {rank}): {rand_eval['reconstruction_error']:.2e}")
    
    # Compare to original low-rank matrix
    full_to_true = torch.norm(A - U_full[:, :rank] @ torch.diag(S_full[:rank]) @ V_full[:rank, :]).item() / torch.norm(A).item()
    rand_to_true = torch.norm(A - U_rand @ torch.diag(S_rand) @ V_rand.T).item() / torch.norm(A).item()
    
    print("\nReconstruction Error (to true low-rank matrix):")
    print(f"Full SVD (top {rank} components): {full_to_true:.2e}")
    print(f"Randomized SVD (rank {rank}): {rand_to_true:.2e}")
    
    # Compare singular values
    print("\nSingular Values:")
    print("True:", S_true.numpy())
    print("Full SVD:", S_full[:rank].numpy())
    print("Randomized SVD:", S_rand.numpy())
    
    # Plot singular values comparison
    plt.figure(figsize=(10, 6))
    plt.bar([f"True {i+1}" for i in range(rank)], S_true.numpy(), alpha=0.7, label="True")
    plt.bar([f"Full {i+1}" for i in range(rank)], S_full[:rank].numpy(), alpha=0.7, label="Full SVD")
    plt.bar([f"Rand {i+1}" for i in range(rank)], S_rand.numpy(), alpha=0.7, label="Randomized SVD")
    plt.title("Singular Values Comparison")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Examine the effect of oversampling and power iterations
    def study_randomized_parameters():
        """Study the effect of oversampling and power iterations on randomized SVD."""
        oversampling_values = [0, 5, 10, 20]
        power_iterations = [0, 1, 2, 4]
        
        results = np.zeros((len(oversampling_values), len(power_iterations)))
        timings = np.zeros((len(oversampling_values), len(power_iterations)))
        
        for i, p in enumerate(oversampling_values):
            for j, q in enumerate(power_iterations):
                start_time = time.time()
                U_r, S_r, V_r = randomized_svd(A_noisy, k=rank, p=p, q=q)
                timings[i, j] = time.time() - start_time
                
                # Convert V_r from V^T to V
                V_r = V_r.T
                
                # Calculate reconstruction error
                error = torch.norm(A - U_r @ torch.diag(S_r) @ V_r.T).item() / torch.norm(A).item()
                results[i, j] = error
        
        # Plot the results
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        sns.heatmap(results, annot=True, fmt=".2e", cmap="viridis_r",
                   xticklabels=power_iterations, yticklabels=oversampling_values)
        plt.title("Reconstruction Error")
        plt.xlabel("Power Iterations (q)")
        plt.ylabel("Oversampling (p)")
        
        plt.subplot(2, 1, 2)
        sns.heatmap(timings, annot=True, fmt=".4f", cmap="viridis",
                   xticklabels=power_iterations, yticklabels=oversampling_values)
        plt.title("Computation Time (seconds)")
        plt.xlabel("Power Iterations (q)")
        plt.ylabel("Oversampling (p)")
        
        plt.tight_layout()
        plt.show()
        
        return results, timings
    
    # Study the effect of randomized SVD parameters
    param_results, param_timings = study_randomized_parameters()
    
    return A, A_noisy, U_rand, S_rand, V_rand, param_results, param_timings

# Demonstrate randomized SVD
A_rand, A_noisy_rand, U_rand, S_rand, V_rand, param_results, param_timings = demonstrate_randomized_svd()

# %% [markdown]
# ## 5. Comparing SVD Algorithms
# 
# Let's compare the different SVD algorithms in terms of accuracy, speed, and scalability for various types of matrices:

# %%
def compare_svd_algorithms():
    """Compare different SVD algorithms."""
    # Create matrices of different types and sizes
    matrices = {
        "Small Dense (10x8)": create_example_matrix(10, 8, "random"),
        "Medium Dense (100x80)": create_example_matrix(100, 80, "random"),
        "Low Rank (100x80, rank≈10)": create_example_matrix(100, 80, "lowrank"),
        "Image-like (50x30)": create_example_matrix(50, 30, "image")
    }
    
    # Define the algorithms to compare
    algorithms = {
        "Power Method": lambda A: power_method_svd(A),
        "Shifted Power": lambda A: shifted_power_method_svd(A),
        "Bidiagonal": lambda A: bidiagonal_svd(A, full_matrices=False),
        "Jacobi": lambda A: jacobi_svd(A),
        "Randomized": lambda A: randomized_svd(A, k=min(A.shape)-1, p=5, q=2),
        "PyTorch": lambda A: torch.linalg.svd(A, full_matrices=False)
    }
    
    # Collect results
    results = {}
    
    for matrix_name, A in matrices.items():
        results[matrix_name] = {"time": {}, "error": {}, "ortho_U": {}, "ortho_V": {}}
        
        for algo_name, algo_func in algorithms.items():
            try:
                # Skip Jacobi for larger matrices (too slow)
                if algo_name == "Jacobi" and A.shape[0] > 20:
                    results[matrix_name]["time"][algo_name] = float('nan')
                    results[matrix_name]["error"][algo_name] = float('nan')
                    results[matrix_name]["ortho_U"][algo_name] = float('nan')
                    results[matrix_name]["ortho_V"][algo_name] = float('nan')
                    continue
                
                # Measure time
                start_time = time.time()
                U, S, V = algo_func(A)
                results[matrix_name]["time"][algo_name] = time.time() - start_time
                
                # Evaluate results
                if algo_name == "PyTorch":
                    V = V.T  # Convert V
                
                evaluation = evaluate_svd(A, U, S, V)
                results[matrix_name]["error"][algo_name] = evaluation["reconstruction_error"]
                results[matrix_name]["ortho_U"][algo_name] = evaluation["U_orthogonality_error"]
                results[matrix_name]["ortho_V"][algo_name] = evaluation["V_orthogonality_error"]
            except Exception as e:
                print(f"Error with {algo_name} on {matrix_name}: {e}")
                results[matrix_name]["time"][algo_name] = float('nan')
                results[matrix_name]["error"][algo_name] = float('nan')
                results[matrix_name]["ortho_U"][algo_name] = float('nan')
                results[matrix_name]["ortho_V"][algo_name] = float('nan')
    
    # Plot the results
    plt.figure(figsize=(15, 12))
    
    # Plot computation time
    plt.subplot(2, 2, 1)
    for i, matrix_name in enumerate(matrices.keys()):
        times = [results[matrix_name]["time"].get(algo, float('nan')) for algo in algorithms.keys()]
        plt.semilogy([algo for algo in algorithms.keys()], times, 'o-', label=matrix_name)
    
    plt.title("Computation Time")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (seconds, log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    
    # Plot reconstruction error
    plt.subplot(2, 2, 2)
    for i, matrix_name in enumerate(matrices.keys()):
        errors = [results[matrix_name]["error"].get(algo, float('nan')) for algo in algorithms.keys()]
        plt.semilogy([algo for algo in algorithms.keys()], errors, 'o-', label=matrix_name)
    
    plt.title("Reconstruction Error (||A - USV^T||/||A||)")
    plt.xlabel("Algorithm")
    plt.ylabel("Relative Error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot U orthogonality error
    plt.subplot(2, 2, 3)
    for i, matrix_name in enumerate(matrices.keys()):
        ortho_U = [results[matrix_name]["ortho_U"].get(algo, float('nan')) for algo in algorithms.keys()]
        plt.semilogy([algo for algo in algorithms.keys()], ortho_U, 'o-', label=matrix_name)
    
    plt.title("U Orthogonality Error (||U^T U - I||)")
    plt.xlabel("Algorithm")
    plt.ylabel("Error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot V orthogonality error
    plt.subplot(2, 2, 4)
    for i, matrix_name in enumerate(matrices.keys()):
        ortho_V = [results[matrix_name]["ortho_V"].get(algo, float('nan')) for algo in algorithms.keys()]
        plt.semilogy([algo for algo in algorithms.keys()], ortho_V, 'o-', label=matrix_name)
    
    plt.title("V Orthogonality Error (||V^T V - I||)")
    plt.xlabel("Algorithm")
    plt.ylabel("Error (log scale)")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print a summary of the results
    print("Summary of SVD Algorithms Comparison:")
    print("=" * 80)
    
    for matrix_name in matrices.keys():
        print(f"\nMatrix: {matrix_name}")
        print("-" * 80)
        print(f"{'Algorithm':<15} {'Time (s)':<15} {'Reconstruction Error':<25} {'U Orthogonality':<20} {'V Orthogonality':<20}")
        print("-" * 80)
        
        for algo_name in algorithms.keys():
            time_val = results[matrix_name]["time"].get(algo_name, float('nan'))
            error_val = results[matrix_name]["error"].get(algo_name, float('nan'))
            ortho_U_val = results[matrix_name]["ortho_U"].get(algo_name, float('nan'))
            ortho_V_val = results[matrix_name]["ortho_V"].get(algo_name, float('nan'))
            
            print(f"{algo_name:<15} {time_val:<15.6f} {error_val:<25.6e} {ortho_U_val:<20.6e} {ortho_V_val:<20.6e}")
    
    return results

# Compare SVD algorithms
svd_algorithm_comparison = compare_svd_algorithms()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored various algorithms for computing the Singular Value Decomposition (SVD):
# 
# 1. **Power Method**: A simple iterative approach that computes singular values and vectors one at a time. We also implemented a shifted variant for faster convergence.
# 
# 2. **Bidiagonalization Method**: A more efficient two-step approach that first reduces the matrix to bidiagonal form, then computes the SVD of the bidiagonal matrix.
# 
# 3. **Jacobi SVD Algorithm**: An iterative method that directly computes the SVD by applying a sequence of plane rotations to diagonalize the matrix.
# 
# 4. **Randomized SVD**: A modern approach that uses random projections to efficiently compute an approximate SVD, particularly for large matrices.
# 
# 5. **Comparison of Algorithms**: We compared these different algorithms in terms of accuracy, speed, and scalability for various types of matrices.
# 
# Key takeaways:
# 
# - Standard libraries like PyTorch and NumPy typically use highly optimized implementations of the bidiagonalization approach
# - The power method is simple but can be slow for matrices with clustered singular values
# - Jacobi SVD can be accurate but is generally slower for larger matrices
# - Randomized SVD offers excellent performance for large matrices, especially when an approximate low-rank SVD is sufficient
# - The choice of algorithm depends on the specific requirements of the application, including matrix size, structure, and accuracy needs
# 
# In the next notebook, we'll explore practical applications of SVD in various domains.

# %%