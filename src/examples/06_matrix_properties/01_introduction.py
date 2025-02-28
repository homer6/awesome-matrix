# %% [markdown]
# # Matrix Properties: Introduction
# 
# This notebook explores fundamental matrix properties that characterize the behavior and structure of matrices. Understanding these properties is essential for linear algebra applications in various fields like machine learning, computer graphics, and scientific computing.
# 
# We'll focus on the following key properties:
# - Determinant
# - Trace
# - Rank
# - Nullity
# - Condition number
# 
# These properties provide insights into whether a matrix is invertible, how it transforms space, and how numerically stable it is for computational purposes.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

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
# ## Creating Matrices for Examples
# 
# Let's create some example matrices with different properties that we'll use throughout this notebook.

# %%
def create_example_matrices():
    """Create various matrices with different properties."""
    # 1. A simple 2x2 matrix
    A = torch.tensor([[4.0, 3.0], 
                      [2.0, 1.0]])
    
    # 2. A non-invertible (singular) matrix
    B = torch.tensor([[1.0, 2.0], 
                      [2.0, 4.0]])  # Second row is 2 * first row
    
    # 3. A random 3x3 matrix
    C = torch.tensor([[5.0, 7.0, 9.0],
                      [2.0, 3.0, 4.0],
                      [1.0, 8.0, 6.0]])
    
    # 4. An ill-conditioned matrix (almost singular)
    D = torch.tensor([[1.0, 1.0], 
                      [1.0, 1.0001]])  # Almost linearly dependent rows
    
    # 5. A matrix with reduced rank
    E = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 9.0]])  # Third row = 3*first row - 2*second row
    
    # 6. An identity matrix
    I2 = torch.eye(2)
    
    # 7. A diagonal matrix
    F = torch.diag(torch.tensor([3.0, 5.0, 7.0]))
    
    return {"simple_2x2": A, 
            "singular_2x2": B, 
            "random_3x3": C, 
            "ill_conditioned": D, 
            "reduced_rank_3x3": E,
            "identity_2x2": I2,
            "diagonal_3x3": F}

# Create example matrices
matrices = create_example_matrices()

# Display the matrices with nicer formatting
def display_matrix(matrix, title):
    """Display a matrix with a title."""
    print(f"{title}:")
    print(matrix)
    print()

for name, matrix in matrices.items():
    display_matrix(matrix, name.replace('_', ' ').title())
    
# %% [markdown]
# ## Visualizing Matrices
# 
# To better understand matrix properties, let's create some visualization functions to represent matrices graphically.

# %%
def plot_matrix_heatmap(matrix, title="Matrix", annotate=True, cmap=blue_cmap):
    """Plot a matrix as a heatmap with annotations."""
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
    
    plt.tight_layout()
    plt.show()

# Let's visualize some of our matrices
plot_matrix_heatmap(matrices["simple_2x2"], "Simple 2x2 Matrix")
plot_matrix_heatmap(matrices["random_3x3"], "Random 3x3 Matrix")
plot_matrix_heatmap(matrices["diagonal_3x3"], "Diagonal 3x3 Matrix")

# %% [markdown]
# ## Determinant
# 
# The determinant of a square matrix is a scalar value that provides important information about the matrix:
# 
# - If the determinant is non-zero, the matrix is invertible (non-singular)
# - The determinant represents the scaling factor of the linear transformation represented by the matrix
# - Geometrically, the determinant represents the volume scaling factor when the matrix transforms a unit cube
# 
# For a 2×2 matrix $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the determinant is:
# 
# $\det(A) = ad - bc$
# 
# For larger matrices, the determinant can be calculated using cofactor expansion or other methods.

# %%
def calculate_and_display_determinant(matrix_dict):
    """Calculate and display determinants of matrices."""
    results = {}
    
    for name, matrix in matrix_dict.items():
        if matrix.shape[0] == matrix.shape[1]:  # Only square matrices have determinants
            det_numpy = np.linalg.det(matrix.numpy())
            det_torch = torch.linalg.det(matrix).item()
            results[name] = (det_numpy, det_torch)
    
    # Display results
    print("Determinants of Matrices:")
    print("-" * 50)
    print(f"{'Matrix':<20} {'NumPy Determinant':<20} {'PyTorch Determinant':<20}")
    print("-" * 50)
    
    for name, (det_np, det_torch) in results.items():
        print(f"{name.replace('_', ' ').title():<20} {det_np:<20.6f} {det_torch:<20.6f}")

# Calculate determinants for our matrices
calculate_and_display_determinant(matrices)

# %% [markdown]
# ### Geometric Interpretation of Determinant in 2D
# 
# In 2D, the determinant of a matrix represents the area scaling factor when the matrix transforms a unit square. Let's visualize this:

# %%
def plot_determinant_geometric_2d(matrix, title):
    """Visualize the geometric meaning of determinant in 2D."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    
    # Check if it's a 2x2 matrix
    if matrix.shape != (2, 2):
        print("This visualization only works for 2x2 matrices.")
        return
    
    # Vertices of the unit square
    unit_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    
    # Apply the transformation to the unit square
    transformed_square = unit_square @ matrix.T
    
    # Calculate the determinant
    det = np.linalg.det(matrix)
    
    # Plot the original and transformed squares
    plt.figure(figsize=(12, 6))
    
    # Original unit square
    plt.subplot(1, 2, 1)
    plt.fill(unit_square[:, 0], unit_square[:, 1], 'lightblue', alpha=0.5, edgecolor='blue')
    plt.scatter(unit_square[:, 0], unit_square[:, 1], color='blue')
    
    # Add arrows showing basis vectors
    plt.arrow(0, 0, 1, 0, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    plt.arrow(0, 0, 0, 1, head_width=0.05, head_length=0.1, fc='blue', ec='blue')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.title("Original Unit Square\nArea = 1")
    
    # Transformed square
    plt.subplot(1, 2, 2)
    plt.fill(transformed_square[:, 0], transformed_square[:, 1], 'lightcoral', alpha=0.5, edgecolor='red')
    plt.scatter(transformed_square[:, 0], transformed_square[:, 1], color='red')
    
    # Add arrows showing transformed basis vectors
    plt.arrow(0, 0, matrix[0, 0], matrix[1, 0], head_width=0.05, head_length=0.1, fc='red', ec='red')
    plt.arrow(0, 0, matrix[0, 1], matrix[1, 1], head_width=0.05, head_length=0.1, fc='red', ec='red')
    
    # Adjust limits based on the transformed coordinates
    max_coord = max(np.max(np.abs(transformed_square[:, 0])), np.max(np.abs(transformed_square[:, 1])))
    plt.xlim(-max_coord-0.5, max_coord+0.5)
    plt.ylim(-max_coord-0.5, max_coord+0.5)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.title(f"Transformed Square\nDeterminant = {det:.2f}\nArea = |Det| = {abs(det):.2f}")
    
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize determinant for some of our matrices
plot_determinant_geometric_2d(matrices["simple_2x2"], "Geometric Interpretation of Determinant")
plot_determinant_geometric_2d(matrices["singular_2x2"], "Singular Matrix (Zero Determinant)")
plot_determinant_geometric_2d(matrices["identity_2x2"], "Identity Matrix (Determinant = 1)")

# %% [markdown]
# ### Observations:
# 
# 1. **Non-Singular Matrix**: The determinant is non-zero, and the transformed square has a non-zero area.
# 
# 2. **Singular Matrix**: The determinant is zero, and the transformed square collapses into a line, with zero area.
# 
# 3. **Identity Matrix**: The determinant is 1, and the transformation doesn't change the square at all.
# 
# 4. **Negative Determinant**: If the determinant is negative, the transformation includes a reflection, which flips the orientation of the space.
# 
# ## Trace
# 
# The trace of a square matrix is the sum of the elements on the main diagonal. 
# 
# For a matrix $A$, the trace is defined as:
# 
# $\text{Tr}(A) = \sum_{i=1}^{n} A_{ii}$
# 
# The trace has several important properties:
# - It equals the sum of the eigenvalues of the matrix
# - It's invariant under similar transformations: $\text{Tr}(P^{-1}AP) = \text{Tr}(A)$
# - It's a linear operator: $\text{Tr}(A + B) = \text{Tr}(A) + \text{Tr}(B)$

# %%
def calculate_and_display_trace(matrix_dict):
    """Calculate and display traces of matrices."""
    results = {}
    
    for name, matrix in matrix_dict.items():
        if matrix.shape[0] == matrix.shape[1]:  # Only square matrices have traces
            trace_numpy = np.trace(matrix.numpy())
            trace_torch = torch.trace(matrix).item()
            
            # Calculate eigenvalues to verify trace = sum of eigenvalues
            eigenvalues = torch.linalg.eigvals(matrix)
            eigensum = torch.sum(eigenvalues).real.item()
            
            results[name] = (trace_numpy, trace_torch, eigensum)
    
    # Display results
    print("Traces of Matrices:")
    print("-" * 70)
    print(f"{'Matrix':<20} {'Trace':<15} {'Sum of Eigenvalues':<20} {'Equal?':<10}")
    print("-" * 70)
    
    for name, (trace_np, trace_torch, eigensum) in results.items():
        is_equal = np.isclose(trace_torch, eigensum, atol=1e-5)
        print(f"{name.replace('_', ' ').title():<20} {trace_torch:<15.4f} {eigensum:<20.4f} {str(is_equal):<10}")

# Calculate traces for our matrices
calculate_and_display_trace(matrices)

# %% [markdown]
# ### Visualizing the Trace
# 
# Let's highlight the main diagonal elements that contribute to the trace:

# %%
def plot_matrix_with_trace(matrix, title):
    """Plot a matrix heatmap with the diagonal elements highlighted."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
        
    plt.figure(figsize=(8, 6))
    
    # Create a mask for the diagonal elements
    mask = np.zeros_like(matrix, dtype=bool)
    np.fill_diagonal(mask, True)
    
    # Plot the full matrix
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap=blue_cmap, 
                     linewidths=1, cbar=True)
    
    # Highlight the diagonal elements
    for i in range(min(matrix.shape)):
        rect = plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2)
        ax.add_patch(rect)
    
    # Calculate trace
    trace_val = np.trace(matrix)
    
    plt.title(f"{title}\nTrace = {trace_val:.2f}")
    
    # Add row and column indices
    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels([f"Col {i+1}" for i in range(matrix.shape[1])])
    ax.set_yticklabels([f"Row {i+1}" for i in range(matrix.shape[0])])
    
    plt.tight_layout()
    plt.show()

# Visualize trace for some of our matrices
plot_matrix_with_trace(matrices["simple_2x2"], "Simple 2x2 Matrix")
plot_matrix_with_trace(matrices["random_3x3"], "Random 3x3 Matrix")
plot_matrix_with_trace(matrices["diagonal_3x3"], "Diagonal 3x3 Matrix")

# %% [markdown]
# ## Rank and Nullity
# 
# The rank of a matrix is the dimension of the vector space spanned by its columns (or equivalently, its rows). It represents the number of linearly independent columns or rows in the matrix.
# 
# The nullity is the dimension of the null space (kernel) of the matrix, which is the set of all vectors that the matrix maps to zero.
# 
# For an m×n matrix:
# - The maximum possible rank is min(m, n)
# - The Rank-Nullity Theorem states: rank(A) + nullity(A) = n

# %%
def calculate_and_display_rank(matrix_dict):
    """Calculate and display ranks of matrices."""
    results = {}
    
    for name, matrix in matrix_dict.items():
        # Calculate rank
        rank_numpy = np.linalg.matrix_rank(matrix.numpy())
        rank_torch = torch.linalg.matrix_rank(matrix).item()
        
        # Calculate nullity
        n = matrix.shape[1]  # Number of columns
        nullity = n - rank_torch
        
        results[name] = (matrix.shape, rank_torch, nullity)
    
    # Display results
    print("Rank and Nullity of Matrices:")
    print("-" * 75)
    print(f"{'Matrix':<20} {'Shape':<10} {'Rank':<10} {'Nullity':<10} {'Rank + Nullity':<15} {'Columns':<10}")
    print("-" * 75)
    
    for name, (shape, rank, nullity) in results.items():
        print(f"{name.replace('_', ' ').title():<20} {str(shape):<10} {rank:<10} {nullity:<10} {rank+nullity:<15} {shape[1]:<10}")

# Calculate rank and nullity for our matrices
calculate_and_display_rank(matrices)

# %% [markdown]
# ### Visualizing the Rank
# 
# To visualize the rank of a matrix, we can use Singular Value Decomposition (SVD) to show the singular values. The number of non-zero singular values equals the rank of the matrix.

# %%
def plot_matrix_singular_values(matrix, title):
    """Plot the singular values of a matrix to visualize rank."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    
    # Compute SVD
    U, S, Vh = np.linalg.svd(matrix)
    
    # Determine rank based on non-zero singular values
    tol = 1e-10  # Tolerance for considering a singular value as zero
    rank = np.sum(S > tol)
    
    # Plot singular values
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(S) + 1), S, color='skyblue', edgecolor='blue')
    plt.axhline(y=tol, color='r', linestyle='--', label=f'Tolerance: {tol}')
    plt.title(f"Singular Values\nRank = {rank}")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.yscale('log')
    plt.legend()
    
    # Plot the matrix with heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap=blue_cmap, linewidths=1, cbar=True)
    plt.title(f"Matrix (Shape: {matrix.shape})")
    
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize rank for some of our matrices
plot_matrix_singular_values(matrices["simple_2x2"], "Full Rank Matrix")
plot_matrix_singular_values(matrices["singular_2x2"], "Singular Matrix (Reduced Rank)")
plot_matrix_singular_values(matrices["reduced_rank_3x3"], "3x3 Matrix with Reduced Rank")

# %% [markdown]
# ### Column Space and Null Space
# 
# Let's visualize the column space and null space for a simple 2×2 matrix:

# %%
def plot_column_and_null_space(matrix, title):
    """Visualize the column space and null space of a 2x2 matrix."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    
    # Only handle 2x2 matrices for this visualization
    if matrix.shape != (2, 2):
        print("This visualization only works for 2x2 matrices.")
        return
    
    # Calculate rank
    rank = np.linalg.matrix_rank(matrix)
    
    # Extract columns
    col1 = matrix[:, 0]
    col2 = matrix[:, 1]
    
    plt.figure(figsize=(10, 5))
    
    # Plot column space
    plt.subplot(1, 2, 1)
    
    # Draw axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot column vectors
    plt.arrow(0, 0, col1[0], col1[1], head_width=0.2, head_length=0.3, fc='blue', ec='blue', label='Column 1')
    plt.arrow(0, 0, col2[0], col2[1], head_width=0.2, head_length=0.3, fc='red', ec='red', label='Column 2')
    
    # Determine plot limits
    max_val = max(np.max(np.abs(col1)), np.max(np.abs(col2))) * 1.5
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    
    # If rank is 1, shade the column space (line)
    if rank == 1:
        # Find the non-zero column if there is one
        if np.linalg.norm(col1) > 1e-10:
            col = col1
        else:
            col = col2
            
        # Create a line through the origin in the direction of the column
        t = np.linspace(-max_val, max_val, 100)
        length = np.linalg.norm(col)
        if length > 1e-10:  # Avoid division by zero
            direction = col / length
            plt.plot(t * direction[0], t * direction[1], 'g--', alpha=0.5, linewidth=2)
            plt.fill_between([-max_val, max_val], [-max_val, max_val], [-max_val, max_val], 
                             color='green', alpha=0.1)
            plt.text(direction[0] * max_val * 0.8, direction[1] * max_val * 0.8, 
                     "Column Space\n(1D Line)", ha='center', va='center', fontsize=10)
    
    # If rank is 2, shade the column space (plane)
    elif rank == 2:
        plt.fill_between([-max_val, max_val], -max_val, max_val, color='green', alpha=0.1)
        plt.text(0, 0, "Column Space\n(2D Plane = R²)", ha='center', va='center', fontsize=10)
    
    # If rank is 0, just show the origin
    else:
        plt.scatter([0], [0], color='green', s=100, zorder=3)
        plt.text(0, 0.5, "Column Space\n(Origin Only)", ha='center', va='center', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.title(f"Column Space\nRank = {rank}")
    plt.legend()
    
    # Plot null space
    plt.subplot(1, 2, 2)
    
    # Draw axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Determine the null space
    if rank < 2:
        # Calculate the null space vector
        # For a 2x2 matrix, if rank is less than 2, there's a null space vector
        if np.linalg.norm(col1) < 1e-10 and np.linalg.norm(col2) < 1e-10:
            # Both columns are zero, null space is the entire R²
            null_vector1 = np.array([1, 0])
            null_vector2 = np.array([0, 1])
            plt.arrow(0, 0, null_vector1[0], null_vector1[1], head_width=0.2, head_length=0.3, 
                      fc='purple', ec='purple', label='Null Vector 1')
            plt.arrow(0, 0, null_vector2[0], null_vector2[1], head_width=0.2, head_length=0.3, 
                      fc='orange', ec='orange', label='Null Vector 2')
            plt.fill_between([-max_val, max_val], -max_val, max_val, color='purple', alpha=0.1)
            plt.text(0, 0, "Null Space\n(2D Plane = R²)", ha='center', va='center', fontsize=10)
        elif np.abs(np.linalg.det(matrix)) < 1e-10:
            # Matrix is singular, find the null vector
            if np.linalg.norm(col1) > 1e-10:
                # Find a vector perpendicular to col1
                null_vector = np.array([-col1[1], col1[0]])
                if np.dot(null_vector, col2) > 1e-10:
                    null_vector = -null_vector
            else:
                # Find a vector perpendicular to col2
                null_vector = np.array([-col2[1], col2[0]])
            
            # Normalize the null vector
            null_vector = null_vector / np.linalg.norm(null_vector)
            
            plt.arrow(0, 0, null_vector[0], null_vector[1], head_width=0.2, head_length=0.3, 
                      fc='purple', ec='purple', label='Null Vector')
            
            # Create a line through the origin in the direction of the null vector
            t = np.linspace(-max_val, max_val, 100)
            plt.plot(t * null_vector[0], t * null_vector[1], 'm--', alpha=0.5, linewidth=2)
            plt.fill_between([-max_val, max_val], [-max_val, max_val], [-max_val, max_val], 
                             color='purple', alpha=0.1)
            plt.text(null_vector[0] * max_val * 0.8, null_vector[1] * max_val * 0.8, 
                     "Null Space\n(1D Line)", ha='center', va='center', fontsize=10)
    else:
        # Rank is 2, null space is just the zero vector
        plt.scatter([0], [0], color='purple', s=100, zorder=3)
        plt.text(0, 0.5, "Null Space\n(Origin Only)", ha='center', va='center', fontsize=10)
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.title(f"Null Space\nNullity = {2 - rank}")
    plt.legend()
    
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize column and null spaces
plot_column_and_null_space(matrices["simple_2x2"], "Column and Null Spaces (Full Rank)")
plot_column_and_null_space(matrices["singular_2x2"], "Column and Null Spaces (Singular Matrix)")
plot_column_and_null_space(matrices["identity_2x2"], "Column and Null Spaces (Identity Matrix)")

# %% [markdown]
# ## Condition Number
# 
# The condition number of a matrix measures how sensitive the solution of a linear system is to small changes in the input data. It's defined as:
# 
# $\kappa(A) = \|A\| \cdot \|A^{-1}\|$
# 
# Where $\|A\|$ is a matrix norm. For the 2-norm, the condition number equals the ratio of the largest to smallest singular value.
# 
# - A condition number close to 1 indicates a well-conditioned matrix (stable)
# - A large condition number indicates an ill-conditioned matrix (unstable)
# - An infinite condition number indicates a singular matrix

# %%
def calculate_and_display_condition_number(matrix_dict):
    """Calculate and display condition numbers of matrices."""
    results = {}
    
    for name, matrix in matrix_dict.items():
        if matrix.shape[0] == matrix.shape[1]:  # Only for square matrices
            try:
                cond_numpy = np.linalg.cond(matrix.numpy())
                
                # Calculate singular values
                U, S, Vh = np.linalg.svd(matrix.numpy())
                max_sv = S[0]
                min_sv = S[-1]
                
                results[name] = (cond_numpy, max_sv, min_sv)
            except np.linalg.LinAlgError:
                # Handle singular matrices
                results[name] = (float('inf'), "N/A", "N/A")
    
    # Display results
    print("Condition Numbers of Matrices:")
    print("-" * 65)
    print(f"{'Matrix':<20} {'Condition Number':<20} {'Max Singular Value':<20} {'Min Singular Value':<20}")
    print("-" * 65)
    
    for name, (cond, max_sv, min_sv) in results.items():
        max_sv_str = str(max_sv) if isinstance(max_sv, str) else f"{max_sv:.4f}"
        min_sv_str = str(min_sv) if isinstance(min_sv, str) else f"{min_sv:.4f}"
        cond_str = str(cond) if cond == float('inf') else f"{cond:.4f}"
        
        print(f"{name.replace('_', ' ').title():<20} {cond_str:<20} {max_sv_str:<20} {min_sv_str:<20}")

# Calculate condition numbers for our matrices
calculate_and_display_condition_number(matrices)

# %% [markdown]
# ### Visualizing the Condition Number
# 
# Let's visualize how the condition number affects the stability of a linear system by showing how a small change in the input can cause a large change in the output for ill-conditioned matrices.

# %%
def plot_condition_number_effects(matrix, title):
    """Visualize the effects of condition number on stability."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()
    
    # Skip if not a 2x2 matrix
    if matrix.shape != (2, 2):
        print("This visualization only works for 2x2 matrices.")
        return
    
    # Calculate condition number
    try:
        cond = np.linalg.cond(matrix)
    except np.linalg.LinAlgError:
        cond = float('inf')
    
    # Create a circle of unit vectors
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    circle_points = np.column_stack((circle_x, circle_y))
    
    # Apply the transformation to the circle
    try:
        transformed_points = circle_points @ matrix.T
    except:
        print("Error applying transformation.")
        return
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Original unit circle
    plt.subplot(1, 2, 1)
    plt.plot(circle_points[:, 0], circle_points[:, 1], 'b-')
    plt.fill(circle_points[:, 0], circle_points[:, 1], 'lightblue', alpha=0.3)
    
    # Draw axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.title("Unit Circle")
    
    # Transformed circle
    plt.subplot(1, 2, 2)
    plt.plot(transformed_points[:, 0], transformed_points[:, 1], 'r-')
    plt.fill(transformed_points[:, 0], transformed_points[:, 1], 'lightcoral', alpha=0.3)
    
    # Draw axes
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Calculate semi-major and semi-minor axes
    # These correspond to the singular values
    U, S, Vh = np.linalg.svd(matrix)
    
    # Plot singular vectors scaled by singular values
    if not np.isinf(cond):
        for i in range(2):
            v = Vh[i]
            s = S[i]
            plt.arrow(0, 0, s*v[0], s*v[1], head_width=0.1, head_length=0.2, 
                     fc='green', ec='green', width=0.02)
            plt.text(s*v[0]*1.1, s*v[1]*1.1, f"σ{i+1}={s:.2f}", fontsize=10)
    
    # Adjust plot limits based on the transformed circle
    max_coord = max(np.max(np.abs(transformed_points[:, 0])), np.max(np.abs(transformed_points[:, 1]))) * 1.2
    plt.xlim(-max_coord, max_coord)
    plt.ylim(-max_coord, max_coord)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    cond_text = str(cond) if np.isinf(cond) else f"{cond:.2f}"
    plt.title(f"Transformed Circle\nCondition Number = {cond_text}")
    
    plt.suptitle(title, y=1.05, fontsize=16)
    plt.tight_layout()
    plt.show()

# Visualize condition number effects
plot_condition_number_effects(matrices["simple_2x2"], "Well-Conditioned Matrix")
plot_condition_number_effects(matrices["singular_2x2"], "Singular Matrix (Infinite Condition Number)")
plot_condition_number_effects(matrices["ill_conditioned"], "Ill-Conditioned Matrix")
plot_condition_number_effects(matrices["identity_2x2"], "Identity Matrix (Condition Number = 1)")

# %% [markdown]
# ### Key Insights on Condition Number:
# 
# 1. **Well-Conditioned Matrix**: Transforms the unit circle into an ellipse with reasonably balanced semi-major and semi-minor axes.
# 
# 2. **Ill-Conditioned Matrix**: Transforms the unit circle into a very elongated ellipse, indicating that some directions are stretched much more than others.
# 
# 3. **Singular Matrix**: Collapses the unit circle into a line segment, demonstrating the loss of a dimension.
# 
# 4. **Identity Matrix**: Preserves the unit circle perfectly, with a condition number of exactly 1.
# 
# ## Conclusion
# 
# In this notebook, we've explored key matrix properties:
# 
# - **Determinant**: Measures volume scaling and determines invertibility
# - **Trace**: Sum of diagonal elements, equal to the sum of eigenvalues
# - **Rank**: Number of linearly independent rows/columns
# - **Nullity**: Dimension of the null space (vectors mapped to zero)
# - **Condition Number**: Measures sensitivity to perturbations
# 
# These properties provide crucial information about matrix behavior in linear transformations and numerical computations. In the next notebook, we'll explore special types of matrices with unique properties and structures.