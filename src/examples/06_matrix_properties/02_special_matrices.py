# %% [markdown]
# # Special Matrices
# 
# In this notebook, we'll explore special types of matrices that have unique properties and structures. These special matrices appear frequently in various applications, from physics and engineering to computer graphics and data science.
# 
# We'll focus on the following types of special matrices:
# 
# - Symmetric and Skew-Symmetric Matrices
# - Orthogonal and Unitary Matrices
# - Diagonal and Triangular Matrices
# - Toeplitz and Circulant Matrices
# - Sparse Matrices
# 
# Understanding these special matrices helps simplify complex operations and can significantly improve computational efficiency.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg as spla
from scipy import sparse
from scipy.sparse import csr_matrix
import networkx as nx

# For better looking plots
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a custom colormap (light blue to dark blue)
colors = [(0.95, 0.95, 1), (0.0, 0.2, 0.6)]  # light blue to dark blue
blue_cmap = LinearSegmentedColormap.from_list('CustomBlue', colors, N=100)

# %% [markdown]
# ## Visualization Helper Functions
# 
# Let's define some helper functions for visualizing matrices.

# %%
def plot_matrix_heatmap(matrix, title="Matrix", annotate=True, cmap=blue_cmap, highlight_pattern=None):
    """Plot a matrix as a heatmap with optional annotations and pattern highlighting."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
        
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, annot=annotate, fmt=".2f", cmap=cmap, 
                    linewidths=1, cbar=True)
    plt.title(title)
    
    # Add row and column indices
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels([f"Col {i+1}" for i in range(matrix.shape[1])])
    ax.set_yticklabels([f"Row {i+1}" for i in range(matrix.shape[0])])
    
    # Highlight specific pattern if provided
    if highlight_pattern is not None:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if highlight_pattern(i, j, matrix):
                    rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2)
                    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.show()

def create_matrix_property_checker(matrix_constructor, property_name, property_checker):
    """Create and check a matrix for a specific property."""
    matrix = matrix_constructor()
    is_property = property_checker(matrix)
    
    print(f"Matrix is {property_name}: {is_property}")
    print("Matrix:")
    print(matrix)
    
    return matrix

# %% [markdown]
# ## 1. Symmetric and Skew-Symmetric Matrices
# 
# ### Symmetric Matrices
# 
# A symmetric matrix is a square matrix that is equal to its transpose: $A = A^T$.
# 
# Properties of symmetric matrices:
# - All eigenvalues are real (not complex)
# - Eigenvectors corresponding to distinct eigenvalues are orthogonal
# - Can be diagonalized by an orthogonal matrix of eigenvectors

# %%
def create_symmetric_matrix(n=4):
    """Create a random symmetric matrix."""
    # Create a random matrix
    A = torch.rand(n, n)
    # Make it symmetric: A = (A + A^T)/2
    A_symmetric = (A + A.T) / 2
    return A_symmetric

def is_symmetric(matrix, tol=1e-6):
    """Check if a matrix is symmetric."""
    if isinstance(matrix, torch.Tensor):
        return torch.allclose(matrix, matrix.T, atol=tol)
    else:
        return np.allclose(matrix, matrix.T, atol=tol)

# Create and check a symmetric matrix
symmetric_matrix = create_matrix_property_checker(
    lambda: create_symmetric_matrix(4), 
    "symmetric", 
    is_symmetric
)

# Visualize the symmetric matrix
plot_matrix_heatmap(
    symmetric_matrix, 
    "Symmetric Matrix", 
    highlight_pattern=lambda i, j, m: True  # Highlight the entire matrix
)

# %% [markdown]
# ### Eigendecomposition of Symmetric Matrices
# 
# Let's verify that the eigenvalues of a symmetric matrix are real and the eigenvectors are orthogonal.

# %%
def analyze_eigendecomposition(matrix, title):
    """Analyze and visualize the eigendecomposition of a matrix."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix_np)
    
    # Check if eigenvalues are real (for symmetric matrices they should be)
    is_real = np.allclose(eigenvalues.imag, np.zeros_like(eigenvalues.imag), atol=1e-10)
    
    # Check orthogonality of eigenvectors
    orthogonality_matrix = eigenvectors.T @ eigenvectors
    is_orthogonal = np.allclose(orthogonality_matrix, np.eye(len(eigenvalues)), atol=1e-10)
    
    # Display results
    print(f"Eigendecomposition of {title}:")
    print(f"Eigenvalues: {eigenvalues.real}")
    print(f"Eigenvalues are real: {is_real}")
    print(f"Eigenvectors are orthogonal: {is_orthogonal}")
    
    # Plot eigenvalues
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues.real, color='skyblue', edgecolor='blue')
    plt.title("Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    
    # Plot orthogonality matrix (should be identity for orthogonal eigenvectors)
    plt.subplot(1, 2, 2)
    sns.heatmap(orthogonality_matrix.real, annot=True, fmt=".2f", cmap=blue_cmap)
    plt.title("Eigenvector Orthogonality\n(Should be Identity Matrix)")
    
    plt.suptitle(f"Eigendecomposition of {title}", y=1.05)
    plt.tight_layout()
    plt.show()

# Analyze the eigendecomposition of the symmetric matrix
analyze_eigendecomposition(symmetric_matrix, "Symmetric Matrix")

# %% [markdown]
# ### Skew-Symmetric Matrices
# 
# A skew-symmetric matrix is a square matrix whose transpose equals its negative: $A = -A^T$.
# 
# Properties of skew-symmetric matrices:
# - The diagonal elements are always zero
# - All eigenvalues are either zero or pure imaginary
# - If $n$ is odd, a skew-symmetric matrix must have at least one zero eigenvalue

# %%
def create_skew_symmetric_matrix(n=4):
    """Create a random skew-symmetric matrix."""
    # Create a random matrix
    A = torch.rand(n, n)
    # Make it skew-symmetric: A = (A - A^T)/2
    A_skew_symmetric = (A - A.T) / 2
    return A_skew_symmetric

def is_skew_symmetric(matrix, tol=1e-6):
    """Check if a matrix is skew-symmetric."""
    if isinstance(matrix, torch.Tensor):
        return torch.allclose(matrix, -matrix.T, atol=tol)
    else:
        return np.allclose(matrix, -matrix.T, atol=tol)

# Create and check a skew-symmetric matrix
skew_symmetric_matrix = create_matrix_property_checker(
    lambda: create_skew_symmetric_matrix(4), 
    "skew-symmetric", 
    is_skew_symmetric
)

# Visualize the skew-symmetric matrix
plot_matrix_heatmap(
    skew_symmetric_matrix, 
    "Skew-Symmetric Matrix", 
    highlight_pattern=lambda i, j, m: i == j  # Highlight diagonal elements (should be zero)
)

# Analyze the eigendecomposition of the skew-symmetric matrix
analyze_eigendecomposition(skew_symmetric_matrix, "Skew-Symmetric Matrix")

# %% [markdown]
# ## 2. Orthogonal and Unitary Matrices
# 
# ### Orthogonal Matrices
# 
# An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors (orthonormal vectors). This means:
# 
# $Q^T Q = Q Q^T = I$
# 
# Properties of orthogonal matrices:
# - The determinant is either +1 or -1
# - They preserve angles and distances when used as transformations
# - They represent rotations (and possibly reflections) in Euclidean space

# %%
def create_orthogonal_matrix(n=3):
    """Create a random orthogonal matrix using QR decomposition."""
    # Create a random matrix
    A = torch.randn(n, n)
    # Use QR decomposition to get an orthogonal matrix Q
    Q, R = torch.linalg.qr(A)
    return Q

def is_orthogonal(matrix, tol=1e-6):
    """Check if a matrix is orthogonal."""
    if isinstance(matrix, torch.Tensor):
        identity = torch.eye(matrix.shape[0], device=matrix.device)
        return torch.allclose(matrix.T @ matrix, identity, atol=tol) and \
               torch.allclose(matrix @ matrix.T, identity, atol=tol)
    else:
        identity = np.eye(matrix.shape[0])
        return np.allclose(matrix.T @ matrix, identity, atol=tol) and \
               np.allclose(matrix @ matrix.T, identity, atol=tol)

# Create and check an orthogonal matrix
orthogonal_matrix = create_matrix_property_checker(
    lambda: create_orthogonal_matrix(3), 
    "orthogonal", 
    is_orthogonal
)

# Visualize the orthogonal matrix
plot_matrix_heatmap(orthogonal_matrix, "Orthogonal Matrix")

# %% [markdown]
# ### Visualizing the Effect of Orthogonal Matrices
# 
# Orthogonal matrices preserve lengths and angles when used as linear transformations. Let's visualize this property.

# %%
def plot_orthogonal_transformation(matrix, title="Orthogonal Transformation"):
    """Visualize how an orthogonal matrix preserves lengths and angles."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    
    # Skip if not a 2x2 matrix
    if matrix.shape != (2, 2):
        # Extract a 2x2 submatrix for visualization
        matrix = matrix[:2, :2]
    
    # Create a unit circle with rays to visualize angles
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_points = np.column_stack((circle_x, circle_y))
    
    # Create rays from origin to points on circle
    n_rays = 12
    ray_theta = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    ray_points = np.column_stack((np.cos(ray_theta), np.sin(ray_theta)))
    
    # Apply the transformation to the circle and rays
    transformed_circle = circle_points @ matrix.T
    transformed_rays = ray_points @ matrix.T
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Original unit circle and rays
    plt.subplot(1, 2, 1)
    plt.plot(circle_points[:, 0], circle_points[:, 1], 'b-')
    plt.fill(circle_points[:, 0], circle_points[:, 1], 'lightblue', alpha=0.3)
    
    # Draw rays
    for i in range(n_rays):
        plt.plot([0, ray_points[i, 0]], [0, ray_points[i, 1]], 'b-', alpha=0.5)
    
    # Draw axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.title("Original Unit Circle")
    
    # Transformed circle and rays
    plt.subplot(1, 2, 2)
    plt.plot(transformed_circle[:, 0], transformed_circle[:, 1], 'r-')
    plt.fill(transformed_circle[:, 0], transformed_circle[:, 1], 'lightcoral', alpha=0.3)
    
    # Draw transformed rays
    for i in range(n_rays):
        plt.plot([0, transformed_rays[i, 0]], [0, transformed_rays[i, 1]], 'r-', alpha=0.5)
    
    # Draw axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # Calculate and display determinant
    det = np.linalg.det(matrix[:2, :2])
    plt.title(f"Transformed Circle\nDeterminant = {det:.2f}")
    
    # Check if it's a rotation or reflection
    if det > 0:
        transformation_type = "Rotation"
    else:
        transformation_type = "Rotation + Reflection"
    
    plt.suptitle(f"{title}\n({transformation_type})", y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()

# Create a 2x2 orthogonal matrix
orthogonal_matrix_2d = create_orthogonal_matrix(2)

# Visualize the orthogonal transformation
plot_orthogonal_transformation(orthogonal_matrix_2d, "Orthogonal Transformation")

# %% [markdown]
# ## 3. Diagonal and Triangular Matrices
# 
# ### Diagonal Matrices
# 
# A diagonal matrix is a matrix in which only the entries along the main diagonal are non-zero.
# 
# Properties of diagonal matrices:
# - Simple to compute powers and inverse (just apply the operation to each diagonal element)
# - Eigenvalues are precisely the diagonal elements
# - Eigenvectors are the standard basis vectors

# %%
def create_diagonal_matrix(values=None, n=4):
    """Create a diagonal matrix with specified values or random values."""
    if values is None:
        values = torch.rand(n) * 10  # Random values between 0 and 10
    return torch.diag(values)

def is_diagonal(matrix, tol=1e-6):
    """Check if a matrix is diagonal."""
    if isinstance(matrix, torch.Tensor):
        # Create a mask for the diagonal elements
        mask = torch.eye(matrix.shape[0], matrix.shape[1], dtype=torch.bool, device=matrix.device)
        # Check if all off-diagonal elements are close to zero
        return torch.allclose(matrix * (~mask), torch.zeros_like(matrix), atol=tol)
    else:
        # Create a mask for the diagonal elements
        mask = np.eye(matrix.shape[0], matrix.shape[1], dtype=bool)
        # Check if all off-diagonal elements are close to zero
        return np.allclose(matrix * (~mask), np.zeros_like(matrix), atol=tol)

# Create and check a diagonal matrix
diagonal_matrix = create_matrix_property_checker(
    lambda: create_diagonal_matrix(n=4), 
    "diagonal", 
    is_diagonal
)

# Visualize the diagonal matrix
plot_matrix_heatmap(
    diagonal_matrix, 
    "Diagonal Matrix", 
    highlight_pattern=lambda i, j, m: i == j  # Highlight diagonal elements
)

# %% [markdown]
# ### Properties of Diagonal Matrices
# 
# Diagonal matrices have special properties that make them easy to work with. Let's verify some of these properties.

# %%
def demonstrate_diagonal_properties(matrix):
    """Demonstrate some special properties of diagonal matrices."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
    
    # Property 1: Powers are easy to compute
    power = 3
    power_by_elements = np.diag(np.power(np.diag(matrix_np), power))
    power_by_matrix = np.linalg.matrix_power(matrix_np, power)
    
    print(f"Property 1: The {power}rd power of a diagonal matrix")
    print("By raising diagonal elements to power:")
    print(power_by_elements)
    print("By matrix power:")
    print(power_by_matrix)
    print(f"Results match: {np.allclose(power_by_elements, power_by_matrix)}")
    print()
    
    # Property 2: Eigenvalues are the diagonal elements
    eigenvalues = np.linalg.eigvals(matrix_np)
    diagonal_elements = np.diag(matrix_np)
    
    print("Property 2: Eigenvalues are the diagonal elements")
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Diagonal elements: {diagonal_elements}")
    print(f"Results match: {np.allclose(np.sort(eigenvalues), np.sort(diagonal_elements))}")
    print()
    
    # Property 3: Matrix operations with diagonal matrices are simple
    D1 = matrix_np
    D2 = np.diag(np.random.rand(matrix_np.shape[0]) * 5)
    
    # Addition
    sum_by_elements = np.diag(np.diag(D1) + np.diag(D2))
    sum_by_matrix = D1 + D2
    
    print("Property 3: Addition of diagonal matrices")
    print("By adding diagonal elements:")
    print(sum_by_elements)
    print("By matrix addition:")
    print(sum_by_matrix)
    print(f"Results match: {np.allclose(sum_by_elements, sum_by_matrix)}")
    print()
    
    # Multiplication
    mult_by_elements = np.diag(np.diag(D1) * np.diag(D2))
    mult_by_matrix = D1 @ D2
    
    print("Property 4: Multiplication of diagonal matrices")
    print("By multiplying diagonal elements:")
    print(mult_by_elements)
    print("By matrix multiplication:")
    print(mult_by_matrix)
    print(f"Results match: {np.allclose(mult_by_elements, mult_by_matrix)}")

# Demonstrate properties of diagonal matrices
demonstrate_diagonal_properties(diagonal_matrix)

# %% [markdown]
# ### Triangular Matrices
# 
# A triangular matrix is a square matrix where all entries either above or below the main diagonal are zero.
# 
# - **Upper triangular**: All entries below the main diagonal are zero
# - **Lower triangular**: All entries above the main diagonal are zero
# 
# Properties of triangular matrices:
# - The determinant is the product of the diagonal elements
# - Eigenvalues are the diagonal elements
# - The product of two upper (or lower) triangular matrices is also upper (or lower) triangular

# %%
def create_upper_triangular_matrix(n=4):
    """Create a random upper triangular matrix."""
    matrix = torch.rand(n, n)
    # Zero out elements below the diagonal
    mask = torch.tril(torch.ones((n, n)), diagonal=-1) == 1
    matrix[mask] = 0
    return matrix

def create_lower_triangular_matrix(n=4):
    """Create a random lower triangular matrix."""
    matrix = torch.rand(n, n)
    # Zero out elements above the diagonal
    mask = torch.triu(torch.ones((n, n)), diagonal=1) == 1
    matrix[mask] = 0
    return matrix

def is_upper_triangular(matrix, tol=1e-6):
    """Check if a matrix is upper triangular."""
    if isinstance(matrix, torch.Tensor):
        # Create a mask for elements below the diagonal
        mask = torch.tril(torch.ones(matrix.shape, dtype=torch.bool, device=matrix.device), diagonal=-1)
        # Check if all elements below the diagonal are close to zero
        return torch.allclose(matrix[mask], torch.zeros_like(matrix)[mask], atol=tol)
    else:
        # Create a mask for elements below the diagonal
        mask = np.tril(np.ones(matrix.shape, dtype=bool), -1)
        # Check if all elements below the diagonal are close to zero
        return np.allclose(matrix[mask], np.zeros_like(matrix)[mask], atol=tol)

def is_lower_triangular(matrix, tol=1e-6):
    """Check if a matrix is lower triangular."""
    if isinstance(matrix, torch.Tensor):
        # Create a mask for elements above the diagonal
        mask = torch.triu(torch.ones(matrix.shape, dtype=torch.bool, device=matrix.device), diagonal=1)
        # Check if all elements above the diagonal are close to zero
        return torch.allclose(matrix[mask], torch.zeros_like(matrix)[mask], atol=tol)
    else:
        # Create a mask for elements above the diagonal
        mask = np.triu(np.ones(matrix.shape, dtype=bool), 1)
        # Check if all elements above the diagonal are close to zero
        return np.allclose(matrix[mask], np.zeros_like(matrix)[mask], atol=tol)

# Create and check an upper triangular matrix
upper_triangular_matrix = create_matrix_property_checker(
    lambda: create_upper_triangular_matrix(4), 
    "upper triangular", 
    is_upper_triangular
)

# Create and check a lower triangular matrix
lower_triangular_matrix = create_matrix_property_checker(
    lambda: create_lower_triangular_matrix(4), 
    "lower triangular", 
    is_lower_triangular
)

# Visualize the triangular matrices
plot_matrix_heatmap(
    upper_triangular_matrix, 
    "Upper Triangular Matrix", 
    highlight_pattern=lambda i, j, m: j >= i  # Highlight upper triangular part
)

plot_matrix_heatmap(
    lower_triangular_matrix, 
    "Lower Triangular Matrix", 
    highlight_pattern=lambda i, j, m: i >= j  # Highlight lower triangular part
)

# %% [markdown]
# ## 4. Toeplitz and Circulant Matrices
# 
# ### Toeplitz Matrices
# 
# A Toeplitz matrix is a matrix in which each descending diagonal from left to right has constant values. In other words, the matrix elements $A_{i,j}$ depend only on the difference between $i$ and $j$.
# 
# Toeplitz matrices arise in signal processing and time series analysis, where they represent convolution operations with finite-length signals.

# %%
def create_toeplitz_matrix(n=5):
    """Create a random Toeplitz matrix."""
    # Generate 2n-1 values for the diagonals
    diagonals = torch.rand(2 * n - 1) * 10 - 5  # Random values between -5 and 5
    
    # Use scipy to create the Toeplitz matrix
    toeplitz_np = spla.toeplitz(diagonals[n-1:], diagonals[n-1::-1])
    
    # Convert back to PyTorch
    return torch.tensor(toeplitz_np, dtype=torch.float32)

def is_toeplitz(matrix, tol=1e-6):
    """Check if a matrix is a Toeplitz matrix."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
    
    # Check each diagonal for constancy
    n, m = matrix_np.shape
    for k in range(-(n-1), m):
        diagonal = np.diagonal(matrix_np, offset=k)
        if len(diagonal) > 1 and not np.allclose(diagonal, diagonal[0] * np.ones_like(diagonal), atol=tol):
            return False
    return True

# Create and check a Toeplitz matrix
toeplitz_matrix = create_matrix_property_checker(
    lambda: create_toeplitz_matrix(5), 
    "Toeplitz", 
    is_toeplitz
)

# Visualize the Toeplitz matrix
plot_matrix_heatmap(
    toeplitz_matrix, 
    "Toeplitz Matrix", 
    highlight_pattern=lambda i, j, m: i - j == 1  # Highlight one of the diagonals
)

# %% [markdown]
# ### Circulant Matrices
# 
# A circulant matrix is a special type of Toeplitz matrix where each row is a cyclic shift of the row above it. It is completely defined by its first row.
# 
# Circulant matrices have special properties related to the Discrete Fourier Transform (DFT) and are diagonalizable by the DFT matrix.

# %%
def create_circulant_matrix(n=5):
    """Create a random circulant matrix."""
    # Generate n values for the first row
    first_row = torch.rand(n) * 10 - 5  # Random values between -5 and 5
    
    # Use scipy to create the circulant matrix
    circulant_np = spla.circulant(first_row.numpy())
    
    # Convert back to PyTorch
    return torch.tensor(circulant_np, dtype=torch.float32)

def is_circulant(matrix, tol=1e-6):
    """Check if a matrix is a circulant matrix."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
    
    # Check if each row is a cyclic shift of the first row
    n = matrix_np.shape[0]
    first_row = matrix_np[0]
    
    for i in range(1, n):
        # The ith row should be the first row cyclically shifted by i positions
        if not np.allclose(matrix_np[i], np.roll(first_row, i), atol=tol):
            return False
    return True

# Create and check a circulant matrix
circulant_matrix = create_matrix_property_checker(
    lambda: create_circulant_matrix(5), 
    "circulant", 
    is_circulant
)

# Visualize the circulant matrix
plot_matrix_heatmap(
    circulant_matrix, 
    "Circulant Matrix", 
    highlight_pattern=lambda i, j, m: (i + 1) % m.shape[0] == j  # Highlight the super-diagonal and corner element
)

# %% [markdown]
# ### Eigendecomposition of Circulant Matrices
# 
# A key property of circulant matrices is that they can be diagonalized by the Discrete Fourier Transform (DFT) matrix. Let's verify this property.

# %%
def analyze_circulant_eigendecomposition(matrix):
    """Analyze and visualize the eigendecomposition of a circulant matrix."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
    
    n = matrix_np.shape[0]
    
    # Compute the DFT matrix
    # The jk-th element is exp(-2πi*j*k/n)
    dft_matrix = np.zeros((n, n), dtype=complex)
    for j in range(n):
        for k in range(n):
            dft_matrix[j, k] = np.exp(-2j * np.pi * j * k / n)
    
    # Normalize the DFT matrix
    dft_matrix = dft_matrix / np.sqrt(n)
    
    # Compute eigenvalues using the DFT of the first row
    first_row = matrix_np[0]
    eigenvalues_dft = np.fft.fft(first_row)
    
    # Compute eigenvalues and eigenvectors using numpy
    eigenvalues_np, eigenvectors_np = np.linalg.eig(matrix_np)
    
    # Sort the eigenvalues and eigenvectors for comparison
    idx = np.argsort(eigenvalues_np.real)
    eigenvalues_np = eigenvalues_np[idx]
    eigenvectors_np = eigenvectors_np[:, idx]
    
    # Sort the DFT eigenvalues
    idx = np.argsort(eigenvalues_dft.real)
    eigenvalues_dft = eigenvalues_dft[idx]
    
    # Display results
    print("Eigendecomposition of Circulant Matrix:")
    print("Eigenvalues from NumPy:", eigenvalues_np)
    print("Eigenvalues from DFT:", eigenvalues_dft)
    print("Eigenvalues match:", np.allclose(eigenvalues_np, eigenvalues_dft, atol=1e-10))
    
    # Plot comparison of eigenvalues
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(eigenvalues_np.real, eigenvalues_np.imag, color='blue', label='NumPy')
    plt.scatter(eigenvalues_dft.real, eigenvalues_dft.imag, color='red', alpha=0.5, label='DFT')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.title("Eigenvalues in Complex Plane")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot the magnitude of the DFT of the first row
    plt.subplot(1, 2, 2)
    dft_magnitudes = np.abs(np.fft.fft(first_row))
    plt.bar(range(n), dft_magnitudes, color='skyblue', edgecolor='blue')
    plt.title("Magnitude of DFT of First Row\n(= Magnitude of Eigenvalues)")
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    
    plt.suptitle("Eigendecomposition of Circulant Matrix", y=1.05)
    plt.tight_layout()
    plt.show()

# Analyze the eigendecomposition of the circulant matrix
analyze_circulant_eigendecomposition(circulant_matrix)

# %% [markdown]
# ## 5. Sparse Matrices
# 
# Sparse matrices have a large number of zero elements. They are common in many applications, such as network analysis, scientific computing, and machine learning with large datasets.
# 
# Instead of storing all elements, sparse matrices typically store only the non-zero elements along with their positions, which can save significant memory for large matrices.

# %%
def create_sparse_matrix(n=10, density=0.2):
    """Create a random sparse matrix with given density."""
    # Generate a random sparse matrix with given density
    sparse_matrix_np = sparse.random(n, n, density=density, format='csr', dtype=np.float32)
    
    # Convert to a dense matrix for visualization
    return torch.tensor(sparse_matrix_np.toarray(), dtype=torch.float32)

def sparsity_ratio(matrix):
    """Calculate the sparsity ratio (proportion of zero elements)."""
    if isinstance(matrix, torch.Tensor):
        total_elements = matrix.numel()
        zero_elements = (matrix == 0).sum().item()
    else:
        total_elements = matrix.size
        zero_elements = np.sum(matrix == 0)
    
    return zero_elements / total_elements

# Create a sparse matrix
sparse_matrix = create_sparse_matrix(10, density=0.2)
sparsity = sparsity_ratio(sparse_matrix)

print(f"Sparse Matrix (Sparsity: {sparsity:.2f}):")
print(sparse_matrix)

# Visualize the sparse matrix
plot_matrix_heatmap(
    sparse_matrix, 
    f"Sparse Matrix (Sparsity: {sparsity:.2f})", 
    annotate=False  # Turn off annotations for clarity
)

# %% [markdown]
# ### Comparison of Storage Requirements: Dense vs. Sparse
# 
# Let's compare the storage requirements for dense and sparse matrix representations.

# %%
def compare_storage_requirements(matrix, title="Storage Comparison"):
    """Compare storage requirements for dense and sparse representations."""
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.numpy()
    else:
        matrix_np = matrix
    
    # Create sparse matrix representations
    csr = sparse.csr_matrix(matrix_np)
    csc = sparse.csc_matrix(matrix_np)
    coo = sparse.coo_matrix(matrix_np)
    
    # Calculate memory usage
    dense_memory = matrix_np.nbytes
    csr_memory = csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes
    csc_memory = csc.data.nbytes + csc.indptr.nbytes + csc.indices.nbytes
    coo_memory = coo.data.nbytes + coo.row.nbytes + coo.col.nbytes
    
    # Calculate the proportion of non-zero elements
    total_elements = matrix_np.size
    non_zero_elements = np.count_nonzero(matrix_np)
    
    print(f"{title}:")
    print(f"Matrix shape: {matrix_np.shape}")
    print(f"Total elements: {total_elements}")
    print(f"Non-zero elements: {non_zero_elements} ({non_zero_elements/total_elements:.2%})")
    print(f"Dense storage (bytes): {dense_memory}")
    print(f"CSR sparse storage (bytes): {csr_memory} ({csr_memory/dense_memory:.2%} of dense)")
    print(f"CSC sparse storage (bytes): {csc_memory} ({csc_memory/dense_memory:.2%} of dense)")
    print(f"COO sparse storage (bytes): {coo_memory} ({coo_memory/dense_memory:.2%} of dense)")
    
    # Plot the comparison
    plt.figure(figsize=(10, 6))
    
    formats = ['Dense', 'CSR', 'CSC', 'COO']
    memory_usage = [dense_memory, csr_memory, csc_memory, coo_memory]
    
    plt.bar(formats, memory_usage, color=['#2c3e50', '#e74c3c', '#3498db', '#2ecc71'])
    plt.ylabel('Memory Usage (bytes)')
    plt.title(f'Matrix Storage Comparison\n({non_zero_elements}/{total_elements} non-zero elements, {non_zero_elements/total_elements:.2%})')
    
    # Add value labels
    for i, v in enumerate(memory_usage):
        plt.text(i, v + 0.5, f"{v}", ha='center')
    
    plt.tight_layout()
    plt.show()

# Compare storage requirements for our sparse matrix
compare_storage_requirements(sparse_matrix, "Sparse Matrix Storage Comparison")

# %% [markdown]
# ### Real-World Application: Graph Adjacency Matrix
# 
# Sparse matrices are often used to represent graphs, where the adjacency matrix has a non-zero entry at position (i,j) if there is an edge from node i to node j.
# 
# Let's create a small graph and visualize both the graph and its adjacency matrix.

# %%
def create_random_graph(n=10, edge_probability=0.2):
    """Create a random graph with n nodes."""
    # Create a random adjacency matrix
    G = nx.gnp_random_graph(n, edge_probability, directed=True)
    
    # Get the adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)
    
    return G, adjacency_matrix

def plot_graph_and_adjacency(G, adjacency_matrix, title="Graph and Adjacency Matrix"):
    """Plot a graph and its adjacency matrix side by side."""
    plt.figure(figsize=(15, 6))
    
    # Plot the graph
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout
    
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, 
            font_size=10, font_weight='bold', arrowsize=15, width=1.5, 
            edge_color='gray', connectionstyle='arc3,rad=0.1')
    
    plt.title("Graph Visualization")
    
    # Plot the adjacency matrix
    plt.subplot(1, 2, 2)
    plt.imshow(adjacency_matrix, cmap='Blues', interpolation='none')
    
    # Add gridlines
    plt.grid(False)
    for i in range(adjacency_matrix.shape[0] + 1):
        plt.axhline(y=i-0.5, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=i-0.5, color='gray', linestyle='-', alpha=0.3)
    
    # Add labels
    plt.xticks(range(adjacency_matrix.shape[0]))
    plt.yticks(range(adjacency_matrix.shape[0]))
    plt.xlabel("To Node")
    plt.ylabel("From Node")
    plt.title("Adjacency Matrix")
    
    plt.colorbar(label="Edge Weight")
    
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print some graph metrics
    print(f"Graph Metrics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
    
    # Calculate sparsity of adjacency matrix
    total_elements = adjacency_matrix.size
    non_zero_elements = np.count_nonzero(adjacency_matrix)
    sparsity = 1 - (non_zero_elements / total_elements)
    
    print(f"Adjacency Matrix Sparsity: {sparsity:.4f}")

# Create a random graph and its adjacency matrix
random_graph, adj_matrix = create_random_graph(10, edge_probability=0.2)

# Plot the graph and its adjacency matrix
plot_graph_and_adjacency(random_graph, adj_matrix, "Random Graph and Its Adjacency Matrix")

# Compare storage requirements for the adjacency matrix
compare_storage_requirements(adj_matrix, "Graph Adjacency Matrix Storage Comparison")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored various types of special matrices:
# 
# 1. **Symmetric and Skew-Symmetric Matrices**:
#    - Symmetric matrices have real eigenvalues and orthogonal eigenvectors
#    - Skew-symmetric matrices have pure imaginary eigenvalues (or zero)
# 
# 2. **Orthogonal Matrices**:
#    - Preserve lengths and angles in transformations
#    - Represent rotations and reflections
# 
# 3. **Diagonal and Triangular Matrices**:
#    - Simplify many matrix operations
#    - Eigenvalues are the diagonal elements
# 
# 4. **Toeplitz and Circulant Matrices**:
#    - Have constant diagonals
#    - Circulant matrices can be diagonalized by the DFT matrix
# 
# 5. **Sparse Matrices**:
#    - Efficiently store matrices with mostly zero elements
#    - Essential for large-scale applications like graph analysis
# 
# Understanding these special matrices helps in recognizing patterns in data, choosing efficient algorithms, and implementing optimized solutions for various problems in science, engineering, and computing.