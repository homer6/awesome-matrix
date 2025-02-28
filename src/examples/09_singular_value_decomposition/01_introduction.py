# %% [markdown]
# # Singular Value Decomposition: Introduction
# 
# Singular Value Decomposition (SVD) is a fundamental matrix factorization technique with wide-ranging applications in signal processing, statistics, machine learning, and many other fields. It decomposes a matrix into the product of three matrices, revealing important properties of the original matrix.
# 
# In this notebook, we will:
# 
# 1. Understand the concept of SVD and its geometric interpretation
# 2. Implement SVD from scratch
# 3. Visualize the decomposition
# 4. Compare with built-in SVD functions
# 
# SVD is particularly valuable because it works for any matrix (not just square or full-rank matrices) and provides a way to decompose a matrix into its fundamental components.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse

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
# ## Basic Concept of Singular Value Decomposition
# 
# For any matrix $A \in \mathbb{R}^{m \times n}$, the Singular Value Decomposition (SVD) is given by:
# 
# $$A = U \Sigma V^T$$
# 
# where:
# - $U \in \mathbb{R}^{m \times m}$ is an orthogonal matrix whose columns are the left singular vectors of $A$
# - $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing the singular values of $A$ in descending order
# - $V \in \mathbb{R}^{n \times n}$ is an orthogonal matrix whose columns are the right singular vectors of $A$
# 
# The singular values are non-negative real numbers and represent the "strength" or "importance" of the corresponding singular vectors in the decomposition.
# 
# Let's start by creating an example matrix and visualizing its SVD:

# %%
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
    else:
        raise ValueError("Unknown method")
    
    return A

# Create a simple example matrix
A_simple = create_example_matrix(m=3, n=2, method="simple")

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

plot_matrix(A_simple, "Example Matrix A (3×2)")

# %% [markdown]
# ## Computing SVD using Eigendecomposition
# 
# One approach to compute the SVD is through the eigendecomposition of $A^TA$ and $AA^T$:
# 
# 1. The columns of $V$ are the eigenvectors of $A^TA$
# 2. The columns of $U$ are the eigenvectors of $AA^T$
# 3. The singular values in $\Sigma$ are the square roots of the eigenvalues of $A^TA$ (or $AA^T$)
# 
# Let's implement this approach:

# %%
def compute_svd_via_eigen(A):
    """
    Compute SVD using eigendecomposition of A^T A and A A^T.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        U: Left singular vectors
        S: Singular values
        V: Right singular vectors
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    
    # Compute A^T A and A A^T
    ATA = A.T @ A  # n x n
    AAT = A @ A.T  # m x m
    
    # Get eigenvalues and eigenvectors of A^T A
    eigvals_ATA, eigvecs_ATA = torch.linalg.eigh(ATA)
    
    # Get eigenvalues and eigenvectors of A A^T
    eigvals_AAT, eigvecs_AAT = torch.linalg.eigh(AAT)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices_ATA = torch.argsort(eigvals_ATA, descending=True)
    eigvals_ATA = eigvals_ATA[sorted_indices_ATA]
    eigvecs_ATA = eigvecs_ATA[:, sorted_indices_ATA]
    
    sorted_indices_AAT = torch.argsort(eigvals_AAT, descending=True)
    eigvals_AAT = eigvals_AAT[sorted_indices_AAT]
    eigvecs_AAT = eigvecs_AAT[:, sorted_indices_AAT]
    
    # Compute singular values
    # We use eigenvalues from A^T A, but could also use those from A A^T
    S = torch.sqrt(torch.clamp(eigvals_ATA, min=0))
    
    # Set V as eigenvectors of A^T A
    V = eigvecs_ATA
    
    # Set U as eigenvectors of A A^T
    U = eigvecs_AAT
    
    # When A is not square, we need to handle the cases m > n or n > m
    min_dim = min(m, n)
    r = min_dim  # Assuming full rank for simplicity
    
    # If m > n, then we need to compute some columns of U using A V / sigma
    # U[:, :r] = A @ V[:, :r] @ torch.diag(1.0 / S[:r])
    
    # Alternative approach: use the identity A V_i = sigma_i U_i
    for i in range(min_dim):
        if S[i] > 1e-10:  # Check for numerical stability
            u_i = (A @ V[:, i]) / S[i]
            U[:, i] = u_i
    
    # Return the decomposition
    # For a non-square matrix, we need to adjust the dimensions of Sigma
    S_full = torch.zeros(m, n)
    for i in range(min(m, n)):
        if i < len(S):
            S_full[i, i] = S[i]
    
    return U, S_full, V

# Test SVD on the example matrix
U_simple, S_simple, V_simple = compute_svd_via_eigen(A_simple)

# Display the matrices
plot_matrix(U_simple, "Left Singular Vectors (U)")
plot_matrix(S_simple, "Singular Values (Σ)")
plot_matrix(V_simple, "Right Singular Vectors (V)")

# Verify that A = U Σ V^T
reconstructed_A = U_simple @ S_simple @ V_simple.T
plot_matrix(reconstructed_A, "Reconstructed Matrix (U Σ V^T)")

# Calculate reconstruction error
reconstruction_error = torch.norm(A_simple - reconstructed_A).item()
print(f"Reconstruction error: {reconstruction_error:.2e}")

# Verify that U and V are orthogonal
U_orthogonality = U_simple.T @ U_simple
V_orthogonality = V_simple.T @ V_simple

plot_matrix(U_orthogonality, "U^T U (should be identity)")
plot_matrix(V_orthogonality, "V^T V (should be identity)")

# %% [markdown]
# ## Geometric Interpretation of SVD
# 
# SVD has a beautiful geometric interpretation. It can be thought of as a sequence of three transformations:
# 
# 1. $V^T$ rotates the space
# 2. $\Sigma$ scales the space along the coordinate axes
# 3. $U$ rotates the space again
# 
# Let's visualize this for a 2D case:

# %%
def visualize_svd_2d():
    """Visualize the geometric interpretation of SVD in 2D."""
    # Create a 2×2 matrix for visualization
    A = torch.tensor([[3.0, 1.0],
                      [1.0, 2.0]])
    
    # Compute SVD
    U, S, V = torch.linalg.svd(A)
    
    # Make sure S is a diagonal matrix
    S_matrix = torch.zeros_like(A)
    for i in range(min(A.shape)):
        S_matrix[i, i] = S[i]
    
    # Create a set of points on a circle
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_points = np.vstack([circle_x, circle_y]).T
    
    # Convert to PyTorch tensor
    circle_points = torch.tensor(circle_points, dtype=torch.float32)
    
    # Apply the transformations
    step1_points = circle_points  # Original circle
    step2_points = step1_points @ V  # After V^T (transposed for right multiplication)
    step3_points = step2_points @ S_matrix  # After Σ
    step4_points = step3_points @ U.T  # After U (transposed for right multiplication)
    
    # Create the visualization
    plt.figure(figsize=(16, 4))
    
    # Original circle
    plt.subplot(1, 4, 1)
    plt.scatter(step1_points[:, 0], step1_points[:, 1], alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title("Original Circle")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # After V^T transformation
    plt.subplot(1, 4, 2)
    plt.scatter(step2_points[:, 0], step2_points[:, 1], alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title("After V^T (Rotation)")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # After Σ transformation
    plt.subplot(1, 4, 3)
    plt.scatter(step3_points[:, 0], step3_points[:, 1], alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title("After Σ (Scaling)")
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Final result after U transformation
    plt.subplot(1, 4, 4)
    plt.scatter(step4_points[:, 0], step4_points[:, 1], alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title("After U (Final Rotation)")
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.tight_layout()
    plt.show()
    
    # Show the original matrix and its decomposition
    print("Original Matrix A:")
    print(A.numpy())
    print("\nLeft Singular Vectors (U):")
    print(U.numpy())
    print("\nSingular Values (S):")
    print(S.numpy())
    print("\nRight Singular Vectors (V):")
    print(V.numpy())
    
    # Visualize the matrix transformation
    plt.figure(figsize=(10, 5))
    
    # Original unit vectors
    plt.subplot(1, 2, 1)
    plt.arrow(0, 0, 1, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='i')
    plt.arrow(0, 0, 0, 1, head_width=0.1, head_length=0.1, fc='red', ec='red', label='j')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.title("Original Unit Vectors")
    plt.legend()
    
    # Transformed unit vectors
    plt.subplot(1, 2, 2)
    plt.arrow(0, 0, A[0, 0], A[1, 0], head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='A·i')
    plt.arrow(0, 0, A[0, 1], A[1, 1], head_width=0.1, head_length=0.1, fc='red', ec='red', label='A·j')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-2, 4)
    plt.ylim(-2, 4)
    plt.title("Transformed Unit Vectors (A·i and A·j)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the principal axes (singular vectors)
    plt.figure(figsize=(10, 5))
    
    # Draw the transformed unit circle (ellipse)
    plt.axes().set_aspect('equal')
    ellipse = Ellipse((0, 0), 2*S[0], 2*S[1], 
                     angle=np.degrees(np.arctan2(U[1, 0], U[0, 0])),
                     facecolor='none', edgecolor='green', linewidth=2, alpha=0.7)
    plt.gca().add_patch(ellipse)
    
    # Draw the principal axes (singular vectors scaled by singular values)
    plt.arrow(0, 0, S[0]*U[0, 0], S[0]*U[1, 0], head_width=0.1, head_length=0.1, 
             fc='blue', ec='blue', label=f'σ₁·u₁ ({S[0]:.2f})')
    plt.arrow(0, 0, S[1]*U[0, 1], S[1]*U[1, 1], head_width=0.1, head_length=0.1, 
             fc='red', ec='red', label=f'σ₂·u₂ ({S[1]:.2f})')
    
    # Draw original unit vectors transformed by A
    plt.arrow(0, 0, A[0, 0], A[1, 0], head_width=0.1, head_length=0.1, 
             fc='lightblue', ec='lightblue', linestyle='--', alpha=0.7, label='A·i')
    plt.arrow(0, 0, A[0, 1], A[1, 1], head_width=0.1, head_length=0.1, 
             fc='pink', ec='pink', linestyle='--', alpha=0.7, label='A·j')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.title("Principal Axes of the Transformation")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Visualize SVD in 2D
visualize_svd_2d()

# %% [markdown]
# ## Truncated SVD
# 
# One of the most powerful aspects of SVD is that we can create a low-rank approximation of a matrix by keeping only the largest singular values and their corresponding singular vectors. This is known as truncated SVD.
# 
# Given the SVD $A = U \Sigma V^T$, the best rank-$k$ approximation of $A$ is:
# 
# $$A_k = U_k \Sigma_k V_k^T$$
# 
# where $U_k$, $\Sigma_k$, and $V_k$ contain only the first $k$ columns of $U$, the $k \times k$ upper-left block of $\Sigma$, and the first $k$ columns of $V$, respectively.
# 
# Let's implement and visualize truncated SVD:

# %%
def truncated_svd(A, k):
    """
    Compute truncated SVD, keeping only the top k singular values.
    
    Args:
        A: Input matrix as a PyTorch tensor
        k: Number of singular values to keep
        
    Returns:
        U_k: First k columns of U
        S_k: Diagonal matrix with top k singular values
        V_k: First k columns of V
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    # Compute full SVD
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    
    m, n = A.shape
    
    # Keep only the first k components
    U_k = U[:, :k]
    S_k = torch.zeros(k, k)
    for i in range(k):
        S_k[i, i] = S[i]
    V_k = V[:, :k]
    
    return U_k, S_k, V_k

def visualize_truncated_svd():
    """Visualize truncated SVD for a larger matrix."""
    # Create a larger matrix
    m, n = 10, 8
    A = create_example_matrix(m=m, n=n, method="image")
    
    # Display the original matrix
    plot_matrix(A, "Original Matrix")
    
    # Compute SVD
    U, S, V = torch.linalg.svd(A, full_matrices=False)
    
    # Create a diagonal matrix from S
    S_matrix = torch.diag(S)
    
    # Plot the singular values
    plt.figure(figsize=(10, 6))
    plt.plot(S.numpy(), 'o-')
    plt.title("Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(S)))
    plt.show()
    
    # Visualize different ranks of approximation
    ranks = [1, 2, 3, 4, min(m, n)]
    
    plt.figure(figsize=(15, 10))
    
    # Original matrix
    plt.subplot(2, 3, 1)
    plt.imshow(A.numpy(), cmap='viridis')
    plt.title(f"Original Matrix (Rank {min(m, n)})")
    plt.colorbar()
    
    for i, k in enumerate(ranks[:-1]):
        # Compute rank-k approximation
        U_k, S_k, V_k = truncated_svd(A, k)
        A_k = U_k @ S_k @ V_k.T
        
        # Display the approximation
        plt.subplot(2, 3, i+2)
        plt.imshow(A_k.numpy(), cmap='viridis')
        plt.title(f"Rank-{k} Approximation")
        plt.colorbar()
        
        # Calculate and display the approximation error
        error = torch.norm(A - A_k).item() / torch.norm(A).item()
        plt.xlabel(f"Relative Error: {error:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the energy captured by each singular value
    energy = S**2 / torch.sum(S**2)
    cumulative_energy = torch.cumsum(energy, 0)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(energy)), energy.numpy())
    plt.title("Energy Distribution")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Normalized Energy")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(energy)))
    
    plt.subplot(1, 2, 2)
    plt.plot(range(len(cumulative_energy)), cumulative_energy.numpy(), 'o-')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99%')
    plt.title("Cumulative Energy")
    plt.xlabel("Number of Singular Values")
    plt.ylabel("Cumulative Energy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(range(len(cumulative_energy)))
    
    plt.tight_layout()
    plt.show()
    
    # Find the number of singular values needed to capture 90% and 99% of energy
    k_90 = torch.sum(cumulative_energy < 0.9).item() + 1
    k_99 = torch.sum(cumulative_energy < 0.99).item() + 1
    
    print(f"Number of singular values needed to capture 90% of energy: {k_90}")
    print(f"Number of singular values needed to capture 99% of energy: {k_99}")
    
    return A, U, S, V, cumulative_energy

# Visualize truncated SVD
A_trunc, U_trunc, S_trunc, V_trunc, energy_trunc = visualize_truncated_svd()

# %% [markdown]
# ## Comparing SVD Implementations
# 
# Let's compare our eigendecomposition-based SVD with built-in implementations in terms of performance and accuracy:

# %%
def compare_svd_methods(A):
    """
    Compare different SVD implementations.
    
    Args:
        A: Input matrix
        
    Returns:
        Dictionary with timing and accuracy results
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    methods = []
    times = []
    reconstruction_errors = []
    orthogonality_errors_U = []
    orthogonality_errors_V = []
    
    # Method 1: Our eigendecomposition-based SVD
    start_time = time.time()
    U_eigen, S_eigen, V_eigen = compute_svd_via_eigen(A)
    eigen_time = time.time() - start_time
    
    methods.append("Eigendecomposition")
    times.append(eigen_time)
    
    # Calculate errors
    A_reconstructed_eigen = U_eigen @ S_eigen @ V_eigen.T
    reconstruction_errors.append(torch.norm(A - A_reconstructed_eigen).item() / torch.norm(A).item())
    orthogonality_errors_U.append(torch.norm(U_eigen.T @ U_eigen - torch.eye(U_eigen.shape[1])).item())
    orthogonality_errors_V.append(torch.norm(V_eigen.T @ V_eigen - torch.eye(V_eigen.shape[1])).item())
    
    # Method 2: PyTorch SVD
    start_time = time.time()
    U_torch, S_torch, V_torch = torch.linalg.svd(A, full_matrices=True)
    torch_time = time.time() - start_time
    
    methods.append("PyTorch SVD")
    times.append(torch_time)
    
    # Create a diagonal matrix from S
    m, n = A.shape
    S_torch_matrix = torch.zeros(m, n)
    for i in range(min(m, n)):
        S_torch_matrix[i, i] = S_torch[i]
    
    # Calculate errors
    A_reconstructed_torch = U_torch @ S_torch_matrix @ V_torch
    reconstruction_errors.append(torch.norm(A - A_reconstructed_torch).item() / torch.norm(A).item())
    orthogonality_errors_U.append(torch.norm(U_torch.T @ U_torch - torch.eye(U_torch.shape[1])).item())
    orthogonality_errors_V.append(torch.norm(V_torch.T @ V_torch - torch.eye(V_torch.shape[1])).item())
    
    # Method 3: NumPy SVD (via SciPy)
    A_np = A.numpy()
    start_time = time.time()
    U_np, S_np, Vt_np = scipy.linalg.svd(A_np, full_matrices=True)
    np_time = time.time() - start_time
    
    methods.append("NumPy/SciPy SVD")
    times.append(np_time)
    
    # Create a diagonal matrix from S
    S_np_matrix = np.zeros((m, n))
    for i in range(min(m, n)):
        S_np_matrix[i, i] = S_np[i]
    
    # Calculate errors
    A_reconstructed_np = U_np @ S_np_matrix @ Vt_np
    reconstruction_errors.append(np.linalg.norm(A_np - A_reconstructed_np) / np.linalg.norm(A_np))
    orthogonality_errors_U.append(np.linalg.norm(U_np.T @ U_np - np.eye(U_np.shape[1])))
    orthogonality_errors_V.append(np.linalg.norm(Vt_np @ Vt_np.T - np.eye(Vt_np.shape[0])))
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Timing comparison
    plt.subplot(2, 2, 1)
    plt.bar(methods, times)
    plt.ylabel("Time (seconds)")
    plt.title("SVD Computation Time")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Reconstruction error
    plt.subplot(2, 2, 2)
    plt.bar(methods, reconstruction_errors)
    plt.ylabel("Relative Reconstruction Error")
    plt.title("SVD Accuracy (||A - USV^T||/||A||)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Orthogonality error for U
    plt.subplot(2, 2, 3)
    plt.bar(methods, orthogonality_errors_U)
    plt.ylabel("Orthogonality Error")
    plt.title("U Orthogonality (||U^T U - I||)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Orthogonality error for V
    plt.subplot(2, 2, 4)
    plt.bar(methods, orthogonality_errors_V)
    plt.ylabel("Orthogonality Error")
    plt.title("V Orthogonality (||V^T V - I||)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print the results
    print("SVD Method Comparison:")
    print("-" * 80)
    print(f"{'Method':<20} {'Time (s)':<15} {'Reconstruction Error':<25} {'U Orthogonality':<20} {'V Orthogonality':<20}")
    print("-" * 80)
    
    for i, method in enumerate(methods):
        print(f"{method:<20} {times[i]:<15.6f} {reconstruction_errors[i]:<25.6e} {orthogonality_errors_U[i]:<20.6e} {orthogonality_errors_V[i]:<20.6e}")
    
    return {
        'methods': methods,
        'times': times,
        'reconstruction_errors': reconstruction_errors,
        'orthogonality_errors_U': orthogonality_errors_U,
        'orthogonality_errors_V': orthogonality_errors_V
    }

# Create a larger matrix for performance testing
A_large = create_example_matrix(m=100, n=80, method="random")

# Compare SVD methods
svd_comparison = compare_svd_methods(A_large)

# %% [markdown]
# ## SVD for Non-Square and Rank-Deficient Matrices
# 
# One of the strengths of SVD is that it works for any matrix, regardless of shape or rank. Let's explore SVD for non-square and rank-deficient matrices:

# %%
def demonstrate_nonsquare_svd():
    """Demonstrate SVD for non-square matrices."""
    # Create a tall matrix (more rows than columns)
    A_tall = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    
    # Create a wide matrix (more columns than rows)
    A_wide = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0]
    ])
    
    # Create a rank-deficient matrix
    A_deficient = torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],  # 2 × row 1
        [3.0, 6.0, 9.0]   # 3 × row 1
    ])
    
    # Function to visualize SVD
    def visualize_svd(A, name):
        # Compute SVD
        U, S, V = torch.linalg.svd(A, full_matrices=True)
        
        # Create a diagonal matrix from S
        m, n = A.shape
        S_matrix = torch.zeros(m, n)
        for i in range(min(m, n)):
            S_matrix[i, i] = S[i]
        
        # Display matrices
        print(f"SVD of {name} Matrix {A.shape}:")
        
        plot_matrix(A, f"Original {name} Matrix")
        plot_matrix(U, "Left Singular Vectors (U)")
        plot_matrix(S_matrix, "Singular Values (Σ)")
        plot_matrix(V.T, "Right Singular Vectors (V)")
        
        # Reconstruct and check error
        A_reconstructed = U @ S_matrix @ V
        error = torch.norm(A - A_reconstructed).item() / torch.norm(A).item()
        plot_matrix(A_reconstructed, f"Reconstructed Matrix (Error: {error:.2e})")
        
        # Plot singular values
        plt.figure(figsize=(8, 4))
        plt.plot(S.numpy(), 'o-')
        plt.title(f"Singular Values of {name} Matrix")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(S)))
        plt.show()
        
        return U, S, V
    
    # Visualize SVD for each matrix
    results_tall = visualize_svd(A_tall, "Tall")
    results_wide = visualize_svd(A_wide, "Wide")
    results_deficient = visualize_svd(A_deficient, "Rank-Deficient")
    
    # For the rank-deficient matrix, demonstrate the nullspace
    U_def, S_def, V_def = results_deficient
    
    # Find the rank (number of non-zero singular values)
    rank = torch.sum(S_def > 1e-10).item()
    print(f"Rank of the rank-deficient matrix: {rank}")
    
    # The nullspace vectors are the columns of V corresponding to zero singular values
    nullspace_vectors = V_def.T[:, rank:]
    
    if nullspace_vectors.shape[1] > 0:
        print("Nullspace vectors:")
        print(nullspace_vectors.numpy())
        
        # Verify that these vectors are in the nullspace
        for i in range(nullspace_vectors.shape[1]):
            v = nullspace_vectors[:, i]
            Av = A_deficient @ v
            print(f"||A·v{i+1}|| = {torch.norm(Av).item():.2e}")
    else:
        print("No nullspace vectors found (matrix is full rank).")
    
    return A_tall, A_wide, A_deficient, results_tall, results_wide, results_deficient

# Demonstrate SVD for non-square and rank-deficient matrices
A_tall, A_wide, A_deficient, results_tall, results_wide, results_deficient = demonstrate_nonsquare_svd()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored the Singular Value Decomposition (SVD), a fundamental matrix factorization technique:
# 
# 1. We've implemented SVD from scratch using eigendecomposition of $A^TA$ and $AA^T$.
# 
# 2. We've visualized the geometric interpretation of SVD as a sequence of transformations.
# 
# 3. We've demonstrated truncated SVD for low-rank matrix approximation.
# 
# 4. We've compared different SVD implementations in terms of performance and accuracy.
# 
# 5. We've explored SVD for non-square and rank-deficient matrices.
# 
# SVD has many important properties:
# 
# - It works for any matrix, regardless of shape or rank.
# - It provides orthogonal bases for the four fundamental subspaces (column space, row space, nullspace, and left nullspace).
# - It allows us to express a matrix as a sum of rank-1 matrices.
# - It gives the best low-rank approximation to a matrix (via truncated SVD).
# 
# In the next notebook, we'll explore some of the computational aspects of SVD, including more efficient algorithms and numerical considerations.

# %%