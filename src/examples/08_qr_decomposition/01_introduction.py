# %% [markdown]
# # QR Decomposition: Introduction
# 
# QR decomposition is a fundamental matrix factorization method that decomposes a matrix into the product of an orthogonal matrix (Q) and an upper triangular matrix (R). This decomposition has numerous applications in numerical linear algebra, particularly for solving linear systems, finding eigenvalues, and least squares problems.
# 
# In this notebook, we will:
# 
# 1. Understand the concept of QR decomposition
# 2. Implement QR decomposition using the Gram-Schmidt process
# 3. Visualize the decomposition geometrically
# 4. Compare with built-in QR decomposition functions
# 5. Explore the numerical stability of different QR algorithms
# 
# QR decomposition is particularly valuable because the orthogonality of Q preserves vector norms and angles, making it numerically stable for many applications.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg
import time
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
# ## Basic Concept of QR Decomposition
# 
# For a matrix $A \in \mathbb{R}^{m \times n}$ with $m \geq n$, the QR decomposition finds matrices $Q$ and $R$ such that:
# 
# $A = QR$
# 
# where:
# - $Q \in \mathbb{R}^{m \times m}$ is an orthogonal matrix ($Q^T Q = I$)
# - $R \in \mathbb{R}^{m \times n}$ is an upper triangular matrix with zeros below the diagonal
# 
# In the case where $m > n$, we often use the "economy" or "reduced" QR decomposition, where:
# - $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns
# - $R \in \mathbb{R}^{n \times n}$ is square upper triangular
# 
# Let's start by creating an example matrix to decompose:

# %%
def create_example_matrix(m=4, n=3, method="random"):
    """Create a matrix for QR decomposition demonstration."""
    if method == "random":
        # Create a random matrix
        A = torch.rand(m, n) * 10 - 5  # Values between -5 and 5
    elif method == "simple":
        # Simple predefined matrix for clear demonstration
        if m == 3 and n == 2:
            A = torch.tensor([
                [12.0, 6.0],
                [3.0, 4.0],
                [4.0, 8.0]
            ])
        else:
            raise ValueError(f"Simple method doesn't support dimensions {m}x{n}")
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
# ## QR Decomposition using Gram-Schmidt Process
# 
# The classic method for computing QR decomposition is the Gram-Schmidt orthogonalization process, which transforms a set of linearly independent vectors into an orthogonal or orthonormal set.
# 
# Here's how it works:
# 
# 1. Start with columns of matrix $A$: $\{a_1, a_2, ..., a_n\}$
# 2. Compute orthogonal vectors $\{u_1, u_2, ..., u_n\}$:
#    - $u_1 = a_1$
#    - $u_2 = a_2 - \text{proj}_{u_1}(a_2)$
#    - $u_3 = a_3 - \text{proj}_{u_1}(a_3) - \text{proj}_{u_2}(a_3)$
#    - ...
# 3. Normalize to get orthonormal vectors $\{q_1, q_2, ..., q_n\}$:
#    - $q_i = \frac{u_i}{||u_i||}$
# 4. Form the matrix $Q = [q_1, q_2, ..., q_n]$
# 5. Compute $R$ such that $A = QR$, which gives us:
#    - $R_{ij} = q_i^T a_j$ for $i \leq j$
#    - $R_{ij} = 0$ for $i > j$
# 
# Let's implement this process:

# %%
def qr_gram_schmidt(A):
    """
    Compute QR decomposition using the Gram-Schmidt process.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    Q = torch.zeros((m, n), dtype=A.dtype)
    R = torch.zeros((n, n), dtype=A.dtype)
    
    for j in range(n):
        # Get the j-th column vector
        v = A[:, j].clone()
        
        # Orthogonalize with respect to previous columns of Q
        for i in range(j):
            # Project onto previous orthogonal vectors
            R[i, j] = torch.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        
        # Compute the norm of the orthogonalized vector
        R[j, j] = torch.norm(v)
        
        # Normalize to get an orthonormal vector
        if R[j, j] > 1e-10:  # Check for numerical stability
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = torch.zeros(m, dtype=A.dtype)
    
    return Q, R

# Test the QR decomposition on our example matrix
Q_simple, R_simple = qr_gram_schmidt(A_simple)

# Display the results
plot_matrix(Q_simple, "Orthogonal Matrix Q")
plot_matrix(R_simple, "Upper Triangular Matrix R")

# Check that A = QR
A_reconstructed = Q_simple @ R_simple
plot_matrix(A_reconstructed, "Reconstructed Matrix (Q×R)")

# Calculate reconstruction error
reconstruction_error = torch.norm(A_simple - A_reconstructed).item()
print(f"Reconstruction error: {reconstruction_error:.2e}")

# Verify that Q is orthogonal (Q^T Q = I)
Q_orthogonality = Q_simple.T @ Q_simple
plot_matrix(Q_orthogonality, "Q^T Q (should be identity)")

orthogonality_error = torch.norm(Q_orthogonality - torch.eye(Q_orthogonality.shape[0])).item()
print(f"Orthogonality error: {orthogonality_error:.2e}")

# %% [markdown]
# ### Visualizing Gram-Schmidt Orthogonalization
# 
# To better understand how the Gram-Schmidt process works, let's visualize it in a 2D or 3D space. We'll show how each column of A gets transformed into orthogonal vectors:

# %%
def visualize_gram_schmidt_2d():
    """Visualize Gram-Schmidt process in 2D."""
    # Create a simple 2×2 matrix
    A = torch.tensor([[3.0, 2.0],
                      [1.0, 2.0]])
    
    # Extract columns
    a1 = A[:, 0].numpy()
    a2 = A[:, 1].numpy()
    
    # Apply Gram-Schmidt
    u1 = a1.copy()  # First vector is unchanged
    q1 = u1 / np.linalg.norm(u1)  # Normalize
    
    # Orthogonalize second vector
    proj = np.dot(a2, q1) * q1
    u2 = a2 - proj
    q2 = u2 / np.linalg.norm(u2)
    
    # Compute R matrix components
    r11 = np.linalg.norm(u1)
    r12 = np.dot(q1, a2)
    r22 = np.linalg.norm(u2)
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    # Original vectors
    plt.arrow(0, 0, a1[0], a1[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='a1')
    plt.arrow(0, 0, a2[0], a2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='a2')
    
    # Orthogonal vectors
    plt.arrow(0, 0, q1[0], q1[1], head_width=0.2, head_length=0.3, fc='green', ec='green', label='q1')
    plt.arrow(0, 0, q2[0], q2[1], head_width=0.2, head_length=0.3, fc='purple', ec='purple', label='q2')
    
    # Projection
    plt.arrow(0, 0, proj[0], proj[1], head_width=0.2, head_length=0.3, fc='orange', ec='orange', 
              linestyle='dashed', label='proj_q1(a2)')
    
    # Draw the projection line
    plt.plot([a2[0], proj[0]], [a2[1], proj[1]], 'k--', alpha=0.7)
    
    plt.axis('equal')
    plt.xlim(-1, 4)
    plt.ylim(-1, 4)
    plt.title('Gram-Schmidt Process in 2D')
    plt.legend()
    plt.show()
    
    # Show the resulting matrices
    Q = np.column_stack([q1, q2])
    R = np.array([[r11, r12],
                  [0, r22]])
    
    print("Original Matrix A:")
    print(A.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q)
    print("\nUpper Triangular Matrix R:")
    print(R)
    print("\nReconstruction A = QR:")
    print(Q @ R)
    
    return A.numpy(), Q, R

# Visualize Gram-Schmidt in 2D
A_2d, Q_2d, R_2d = visualize_gram_schmidt_2d()

# %% [markdown]
# ### Visualizing Gram-Schmidt in 3D
# 
# Now let's visualize the Gram-Schmidt process for a 3×3 matrix in 3D space:

# %%
def visualize_gram_schmidt_3d():
    """Visualize Gram-Schmidt process in 3D."""
    # Create a simple 3×3 matrix
    A = torch.tensor([[3.0, 1.0, 1.0],
                      [1.0, 2.0, 0.0],
                      [1.0, 1.0, 2.0]])
    
    # Extract columns
    a1 = A[:, 0].numpy()
    a2 = A[:, 1].numpy()
    a3 = A[:, 2].numpy()
    
    # Apply Gram-Schmidt
    u1 = a1.copy()
    q1 = u1 / np.linalg.norm(u1)
    
    # Second vector
    proj1 = np.dot(a2, q1) * q1
    u2 = a2 - proj1
    q2 = u2 / np.linalg.norm(u2)
    
    # Third vector
    proj1_3 = np.dot(a3, q1) * q1
    proj2_3 = np.dot(a3, q2) * q2
    u3 = a3 - proj1_3 - proj2_3
    q3 = u3 / np.linalg.norm(u3)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw original vectors
    ax.quiver(0, 0, 0, a1[0], a1[1], a1[2], color='blue', label='a1')
    ax.quiver(0, 0, 0, a2[0], a2[1], a2[2], color='red', label='a2')
    ax.quiver(0, 0, 0, a3[0], a3[1], a3[2], color='darkred', label='a3')
    
    # Draw orthogonal vectors
    ax.quiver(0, 0, 0, q1[0], q1[1], q1[2], color='green', label='q1')
    ax.quiver(0, 0, 0, q2[0], q2[1], q2[2], color='purple', label='q2')
    ax.quiver(0, 0, 0, q3[0], q3[1], q3[2], color='orange', label='q3')
    
    ax.set_xlim([-1, 3])
    ax.set_ylim([-1, 3])
    ax.set_zlim([-1, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Gram-Schmidt Process in 3D')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Compute R matrix
    r11 = np.linalg.norm(u1)
    r12 = np.dot(q1, a2)
    r13 = np.dot(q1, a3)
    r22 = np.linalg.norm(u2)
    r23 = np.dot(q2, a3)
    r33 = np.linalg.norm(u3)
    
    # Show the resulting matrices
    Q = np.column_stack([q1, q2, q3])
    R = np.array([[r11, r12, r13],
                  [0, r22, r23],
                  [0, 0, r33]])
    
    print("Original Matrix A:")
    print(A.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q)
    print("\nUpper Triangular Matrix R:")
    print(R)
    print("\nReconstruction A = QR:")
    print(Q @ R)
    
    # Calculate orthogonality of Q
    QTQ = Q.T @ Q
    print("\nQ^T Q (should be identity):")
    print(QTQ)
    print(f"Orthogonality error: {np.linalg.norm(QTQ - np.eye(3)):.2e}")
    
    return A.numpy(), Q, R

# Visualize Gram-Schmidt in 3D
A_3d, Q_3d, R_3d = visualize_gram_schmidt_3d()

# %% [markdown]
# ## Modified Gram-Schmidt Algorithm
# 
# The classic Gram-Schmidt process can suffer from numerical instability due to floating-point errors. The modified Gram-Schmidt algorithm provides better numerical stability by orthogonalizing vectors one by one, rather than all at once.
# 
# Here's the implementation:

# %%
def qr_modified_gram_schmidt(A):
    """
    Compute QR decomposition using the modified Gram-Schmidt process.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    Q = torch.zeros((m, n), dtype=A.dtype)
    R = torch.zeros((n, n), dtype=A.dtype)
    
    # Initialize U as a copy of A
    U = A.clone()
    
    for i in range(n):
        # Compute the norm of the i-th column of U
        R[i, i] = torch.norm(U[:, i])
        
        # Normalize to get an orthonormal vector
        if R[i, i] > 1e-10:  # Check for numerical stability
            Q[:, i] = U[:, i] / R[i, i]
        else:
            Q[:, i] = torch.zeros(m, dtype=A.dtype)
        
        # Orthogonalize remaining columns with respect to the i-th column of Q
        for j in range(i+1, n):
            R[i, j] = torch.dot(Q[:, i], U[:, j])
            U[:, j] = U[:, j] - R[i, j] * Q[:, i]
    
    return Q, R

# Test the modified Gram-Schmidt on our example matrix
Q_modified, R_modified = qr_modified_gram_schmidt(A_simple)

# Display the results
plot_matrix(Q_modified, "Orthogonal Matrix Q (Modified G-S)")
plot_matrix(R_modified, "Upper Triangular Matrix R (Modified G-S)")

# Check that A = QR
A_reconstructed_modified = Q_modified @ R_modified
plot_matrix(A_reconstructed_modified, "Reconstructed Matrix (Q×R)")

# Calculate reconstruction error
reconstruction_error_modified = torch.norm(A_simple - A_reconstructed_modified).item()
print(f"Reconstruction error (Modified G-S): {reconstruction_error_modified:.2e}")

# Verify that Q is orthogonal (Q^T Q = I)
Q_orthogonality_modified = Q_modified.T @ Q_modified
plot_matrix(Q_orthogonality_modified, "Q^T Q (should be identity)")

orthogonality_error_modified = torch.norm(Q_orthogonality_modified - torch.eye(Q_orthogonality_modified.shape[0])).item()
print(f"Orthogonality error (Modified G-S): {orthogonality_error_modified:.2e}")

# %% [markdown]
# ## Householder QR Decomposition
# 
# The Householder QR algorithm is another approach for computing QR decomposition. Instead of the Gram-Schmidt process, it uses Householder reflections to progressively zero out elements below the diagonal.
# 
# A Householder reflection is a transformation that reflects a vector about a hyperplane. It's defined by a vector $v$ such that the reflection matrix is:
# 
# $H = I - 2 \frac{vv^T}{v^T v}$
# 
# Let's implement this approach:

# %%
def qr_householder(A):
    """
    Compute QR decomposition using Householder reflections.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    R = A.clone()
    Q = torch.eye(m, dtype=A.dtype)
    
    for k in range(min(m-1, n)):
        # Extract the column we want to transform
        x = R[k:, k]
        
        # Construct the Householder vector
        e1 = torch.zeros_like(x)
        e1[0] = 1.0
        
        alpha = torch.norm(x)
        # Ensure proper sign of alpha to avoid cancellation
        if x[0] < 0:
            alpha = -alpha
        
        u = x - alpha * e1
        v = u / torch.norm(u)  # Normalize
        
        # Apply the Householder reflection to R
        R[k:, k:] = R[k:, k:] - 2.0 * torch.outer(v, torch.matmul(v, R[k:, k:]))
        
        # Apply the Householder reflection to Q
        Q[:, k:] = Q[:, k:] - 2.0 * torch.matmul(Q[:, k:], torch.outer(v, v))
    
    # Q should be the product of all Householder reflections
    Q = Q.T  # Transpose because we've been applying reflections from the right
    
    # Ensure the diagonal of R is positive
    for i in range(min(m, n)):
        if R[i, i] < 0:
            R[i, i:] = -R[i, i:]
            Q[:, i] = -Q[:, i]
    
    return Q, R

# Test the Householder QR on our example matrix
Q_house, R_house = qr_householder(A_simple)

# Display the results
plot_matrix(Q_house, "Orthogonal Matrix Q (Householder)")
plot_matrix(R_house, "Upper Triangular Matrix R (Householder)")

# Check that A = QR
A_reconstructed_house = Q_house @ R_house
plot_matrix(A_reconstructed_house, "Reconstructed Matrix (Q×R)")

# Calculate reconstruction error
reconstruction_error_house = torch.norm(A_simple - A_reconstructed_house).item()
print(f"Reconstruction error (Householder): {reconstruction_error_house:.2e}")

# Verify that Q is orthogonal (Q^T Q = I)
Q_orthogonality_house = Q_house.T @ Q_house
plot_matrix(Q_orthogonality_house, "Q^T Q (should be identity)")

orthogonality_error_house = torch.norm(Q_orthogonality_house - torch.eye(Q_orthogonality_house.shape[0])).item()
print(f"Orthogonality error (Householder): {orthogonality_error_house:.2e}")

# %% [markdown]
# ## Comparing QR Decomposition Methods
# 
# Now let's compare the different QR decomposition methods we've implemented with the built-in functions in terms of:
# 1. Numerical accuracy
# 2. Computational performance
# 3. Orthogonality of Q

# %%
def compare_qr_methods(A):
    """Compare different QR decomposition methods."""
    # Make a copy for consistent results
    A = A.clone()
    
    # Our implementations
    methods = []
    q_matrices = []
    r_matrices = []
    times = []
    reconstruction_errors = []
    orthogonality_errors = []
    
    # Method 1: Classic Gram-Schmidt
    start_time = time.time()
    Q_gs, R_gs = qr_gram_schmidt(A)
    gs_time = time.time() - start_time
    methods.append("Classic Gram-Schmidt")
    q_matrices.append(Q_gs)
    r_matrices.append(R_gs)
    times.append(gs_time)
    
    # Method 2: Modified Gram-Schmidt
    start_time = time.time()
    Q_mgs, R_mgs = qr_modified_gram_schmidt(A)
    mgs_time = time.time() - start_time
    methods.append("Modified Gram-Schmidt")
    q_matrices.append(Q_mgs)
    r_matrices.append(R_mgs)
    times.append(mgs_time)
    
    # Method 3: Householder
    start_time = time.time()
    Q_house, R_house = qr_householder(A)
    house_time = time.time() - start_time
    methods.append("Householder")
    q_matrices.append(Q_house)
    r_matrices.append(R_house)
    times.append(house_time)
    
    # Method 4: NumPy QR (via SciPy)
    A_np = A.numpy()
    start_time = time.time()
    Q_np, R_np = scipy.linalg.qr(A_np, mode='economic')
    np_time = time.time() - start_time
    methods.append("SciPy QR")
    q_matrices.append(torch.from_numpy(Q_np))
    r_matrices.append(torch.from_numpy(R_np))
    times.append(np_time)
    
    # Method 5: PyTorch QR
    start_time = time.time()
    Q_torch, R_torch = torch.linalg.qr(A, mode='reduced')
    torch_time = time.time() - start_time
    methods.append("PyTorch QR")
    q_matrices.append(Q_torch)
    r_matrices.append(R_torch)
    times.append(torch_time)
    
    # Calculate errors
    for i, (method, Q, R) in enumerate(zip(methods, q_matrices, r_matrices)):
        # Reconstruction error
        reconstructed = Q @ R
        reconstruction_errors.append(torch.norm(A - reconstructed).item())
        
        # Orthogonality error
        Q_orth = Q.T @ Q
        orthogonality_errors.append(torch.norm(Q_orth - torch.eye(Q_orth.shape[0])).item())
    
    # Plot the results
    plt.figure(figsize=(15, 10))
    
    # Computation time
    plt.subplot(2, 2, 1)
    plt.bar(methods, times)
    plt.ylabel("Time (seconds)")
    plt.title("Computation Time")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Reconstruction error
    plt.subplot(2, 2, 2)
    plt.bar(methods, reconstruction_errors)
    plt.ylabel("Reconstruction Error")
    plt.title("A - QR (Frobenius norm)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Orthogonality error
    plt.subplot(2, 2, 3)
    plt.bar(methods, orthogonality_errors)
    plt.ylabel("Orthogonality Error")
    plt.title("Q^T Q - I (Frobenius norm)")
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print the performance comparison
    print("QR Decomposition Method Comparison:")
    print("-" * 100)
    print(f"{'Method':<25} {'Time (s)':<15} {'Reconstruction Error':<25} {'Orthogonality Error':<25}")
    print("-" * 100)
    for method, t, rec_err, orth_err in zip(methods, times, reconstruction_errors, orthogonality_errors):
        print(f"{method:<25} {t:<15.6f} {rec_err:<25.2e} {orth_err:<25.2e}")
    
    return methods, times, reconstruction_errors, orthogonality_errors

# Test on our example matrix
methods, times, rec_errors, orth_errors = compare_qr_methods(A_simple)

# Create a larger random matrix to better highlight performance differences
A_random = create_example_matrix(m=100, n=50, method="random")
methods_large, times_large, rec_errors_large, orth_errors_large = compare_qr_methods(A_random)

# %% [markdown]
# ### Numerical Stability Demonstration
# 
# Let's create an example that specifically demonstrates the numerical instability of the classic Gram-Schmidt process compared to the modified version:

# %%
def demonstrate_numerical_stability():
    """Demonstrate numerical instability of classical Gram-Schmidt."""
    # Create a matrix with nearly linearly dependent columns
    m, n = 5, 3
    
    # Start with orthogonal columns
    Q = torch.eye(m, n)
    
    # Apply a condition number to make it ill-conditioned
    kappa = 1e5  # Condition number
    S = torch.diag(torch.tensor([1.0, 1.0/np.sqrt(kappa), 1.0/kappa]))
    
    # Create a matrix with columns that are nearly linearly dependent
    A = Q @ S
    
    # Add a small random perturbation
    A = A + 1e-10 * torch.randn(m, n)
    
    # Apply different QR methods
    Q_gs, R_gs = qr_gram_schmidt(A)
    Q_mgs, R_mgs = qr_modified_gram_schmidt(A)
    Q_house, R_house = qr_householder(A)
    
    # Calculate orthogonality errors
    orth_err_gs = torch.norm(Q_gs.T @ Q_gs - torch.eye(n)).item()
    orth_err_mgs = torch.norm(Q_mgs.T @ Q_mgs - torch.eye(n)).item()
    orth_err_house = torch.norm(Q_house.T @ Q_house - torch.eye(n)).item()
    
    # Calculate reconstruction errors
    rec_err_gs = torch.norm(A - Q_gs @ R_gs).item()
    rec_err_mgs = torch.norm(A - Q_mgs @ R_mgs).item()
    rec_err_house = torch.norm(A - Q_house @ R_house).item()
    
    # Print results
    print("Numerical Stability Comparison:")
    print("-" * 70)
    print(f"Condition number of test matrix: {kappa:.1e}")
    print("-" * 70)
    print(f"{'Method':<25} {'Orthogonality Error':<25} {'Reconstruction Error':<25}")
    print("-" * 70)
    print(f"{'Classic Gram-Schmidt':<25} {orth_err_gs:<25.2e} {rec_err_gs:<25.2e}")
    print(f"{'Modified Gram-Schmidt':<25} {orth_err_mgs:<25.2e} {rec_err_mgs:<25.2e}")
    print(f"{'Householder':<25} {orth_err_house:<25.2e} {rec_err_house:<25.2e}")
    
    # Plot the orthogonality of Q matrices
    methods = ["Classic G-S", "Modified G-S", "Householder"]
    orth_errors = [orth_err_gs, orth_err_mgs, orth_err_house]
    rec_errors = [rec_err_gs, rec_err_mgs, rec_err_house]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(methods, orth_errors)
    plt.title("Orthogonality Error (Q^T Q - I)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(methods, rec_errors)
    plt.title("Reconstruction Error (A - QR)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize orthogonality matrices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(Q_gs.T @ Q_gs, annot=True, fmt=".3f", cmap=blue_cmap)
    plt.title(f"Q^T Q (Classic G-S)\nError: {orth_err_gs:.2e}")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(Q_mgs.T @ Q_mgs, annot=True, fmt=".3f", cmap=blue_cmap)
    plt.title(f"Q^T Q (Modified G-S)\nError: {orth_err_mgs:.2e}")
    
    plt.subplot(1, 3, 3)
    sns.heatmap(Q_house.T @ Q_house, annot=True, fmt=".3f", cmap=blue_cmap)
    plt.title(f"Q^T Q (Householder)\nError: {orth_err_house:.2e}")
    
    plt.tight_layout()
    plt.show()
    
    return A, [Q_gs, Q_mgs, Q_house], [R_gs, R_mgs, R_house]

# Demonstrate numerical stability
A_ill, Q_methods, R_methods = demonstrate_numerical_stability()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we have explored QR decomposition, a fundamental matrix factorization technique:
# 
# 1. We implemented and compared three different QR decomposition algorithms:
#    - Classic Gram-Schmidt process
#    - Modified Gram-Schmidt process
#    - Householder reflections
# 
# 2. We visualized the Gram-Schmidt process geometrically in 2D and 3D to build intuition.
# 
# 3. We demonstrated the numerical stability advantages of the modified Gram-Schmidt and Householder methods over the classic Gram-Schmidt process.
# 
# 4. We compared our implementations with built-in functions from SciPy and PyTorch.
# 
# Key takeaways:
# 
# - Classic Gram-Schmidt is conceptually simple but can suffer from numerical instability
# - Modified Gram-Schmidt provides better numerical stability with the same computational complexity
# - Householder reflections offer the best numerical stability and are commonly used in production libraries
# - Built-in implementations (SciPy, PyTorch) are typically faster and more robust for practical use
# 
# In the next notebook, we'll explore applications of QR decomposition in solving linear systems, least squares problems, and eigenvalue computations.

# %%