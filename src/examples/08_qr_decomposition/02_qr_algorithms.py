# %% [markdown]
# # QR Decomposition: Algorithms and Implementation
# 
# This notebook delves into the algorithmic details of QR decomposition. We examine different methods for computing QR factorization, their properties, and implementation considerations.
# 
# We'll focus on:
# 
# 1. **Classical Gram-Schmidt Algorithm in Detail**
# 2. **Modified Gram-Schmidt Algorithm**
# 3. **Householder Reflections**
# 4. **Givens Rotations**
# 5. **Optimized Implementations**
# 6. **Numerical Stability Analysis**
# 
# Each method has specific advantages, trade-offs, and use cases that we'll explore through both theoretical explanation and practical implementation.

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

# Create a function to measure reconstruction error and orthogonality
def measure_qr_quality(A, Q, R):
    """Measure the quality of a QR decomposition."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    if isinstance(Q, np.ndarray):
        Q = torch.tensor(Q, dtype=torch.float64)
    if isinstance(R, np.ndarray):
        R = torch.tensor(R, dtype=torch.float64)
    
    # Measure reconstruction error
    QR = Q @ R
    reconstruction_error = torch.norm(A - QR).item()
    
    # Measure orthogonality of Q
    QTQ = Q.T @ Q
    orthogonality_error = torch.norm(QTQ - torch.eye(QTQ.shape[0])).item()
    
    return reconstruction_error, orthogonality_error

# %% [markdown]
# ## 1. Classical Gram-Schmidt Algorithm in Detail
# 
# The Gram-Schmidt process is a method for orthonormalizing a set of vectors. In the context of QR decomposition, we apply it to the columns of matrix $A$.
# 
# Here's a detailed step-by-step breakdown of how the classical Gram-Schmidt algorithm works:

# %%
def classical_gram_schmidt_detailed(A):
    """
    Implement classical Gram-Schmidt with detailed step-by-step visualization.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
        steps: Detailed steps for visualization
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    Q = torch.zeros((m, n), dtype=A.dtype)
    R = torch.zeros((n, n), dtype=A.dtype)
    
    # Store intermediate steps for visualization
    steps = []
    
    for j in range(n):
        # Get the j-th column vector
        v = A[:, j].clone()
        
        # Store the original vector and projections
        step_data = {
            'j': j,
            'original_vector': v.clone(),
            'projections': []
        }
        
        # Orthogonalize with respect to previous columns of Q
        for i in range(j):
            # Calculate projection coefficient
            R[i, j] = torch.dot(Q[:, i], A[:, j])
            
            # Calculate projection vector
            projection = R[i, j] * Q[:, i]
            
            # Store projection for visualization
            step_data['projections'].append({
                'i': i,
                'coefficient': R[i, j].item(),
                'vector': projection.clone()
            })
            
            # Subtract projection
            v = v - projection
        
        # Compute the norm of the orthogonalized vector
        R[j, j] = torch.norm(v)
        
        # Store the orthogonalized vector
        step_data['orthogonalized'] = v.clone()
        
        # Normalize to get an orthonormal vector
        if R[j, j] > 1e-10:  # Check for numerical stability
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = torch.zeros(m, dtype=A.dtype)
        
        # Store the normalized vector
        step_data['normalized'] = Q[:, j].clone()
        
        # Add step to list
        steps.append(step_data)
    
    return Q, R, steps

def visualize_gram_schmidt_steps(A, steps, dim=2):
    """Visualize the Gram-Schmidt process step by step."""
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    
    m, n = A.shape
    
    if dim > m:
        print(f"Warning: Can only visualize in {m} dimensions, not {dim}")
        dim = m
    
    if dim == 2:
        # 2D visualization
        plt.figure(figsize=(15, 4 * n))
        
        for j, step in enumerate(steps):
            # Plot original vector
            plt.subplot(n, 3, j*3 + 1)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot original column vector
            original = step['original_vector'].numpy()
            plt.arrow(0, 0, original[0], original[1], head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue', label=f'a{j+1}')
            
            # Plot previous orthonormal vectors
            for i in range(j):
                q_i = steps[i]['normalized'].numpy()
                plt.arrow(0, 0, q_i[0], q_i[1], head_width=0.1, head_length=0.1, 
                         fc='green', ec='green', label=f'q{i+1}')
            
            plt.title(f"Step {j+1}: Original Vector")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            
            # Plot projections
            plt.subplot(n, 3, j*3 + 2)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot original vector
            plt.arrow(0, 0, original[0], original[1], head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue', alpha=0.5, label=f'a{j+1}')
            
            # Plot projections
            cumulative_projection = np.zeros(m)
            for proj in step['projections']:
                projection = proj['vector'].numpy()
                q_i = steps[proj['i']]['normalized'].numpy()
                
                plt.arrow(0, 0, q_i[0], q_i[1], head_width=0.1, head_length=0.1, 
                         fc='green', ec='green', alpha=0.7, label=f'q{proj["i"]+1}')
                
                # Draw projection from origin
                plt.arrow(0, 0, projection[0], projection[1], head_width=0.1, head_length=0.1, 
                         fc='red', ec='red', alpha=0.7, linestyle='dashed', 
                         label=f'proj_{{{proj["i"]+1}}}(a{j+1})')
                
                cumulative_projection += projection
            
            # Draw orthogonalized vector
            orthogonalized = step['orthogonalized'].numpy()
            plt.arrow(0, 0, orthogonalized[0], orthogonalized[1], head_width=0.1, head_length=0.1, 
                     fc='purple', ec='purple', label=f'u{j+1}')
            
            plt.title(f"Step {j+1}: Projections and Orthogonalization")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            
            # Plot normalized vector
            plt.subplot(n, 3, j*3 + 3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Plot final orthonormal vector
            normalized = step['normalized'].numpy()
            plt.arrow(0, 0, normalized[0], normalized[1], head_width=0.1, head_length=0.1, 
                     fc='purple', ec='purple', label=f'q{j+1}')
            
            # Plot previous orthonormal vectors
            for i in range(j):
                q_i = steps[i]['normalized'].numpy()
                plt.arrow(0, 0, q_i[0], q_i[1], head_width=0.1, head_length=0.1, 
                         fc='green', ec='green', alpha=0.7, label=f'q{i+1}')
            
            plt.title(f"Step {j+1}: Normalized Vector")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
        
        plt.tight_layout()
        plt.show()
    
    elif dim == 3:
        # 3D visualization
        for j, step in enumerate(steps):
            fig = plt.figure(figsize=(15, 5))
            
            # Plot original vector
            ax1 = fig.add_subplot(131, projection='3d')
            
            # Plot original column vector
            original = step['original_vector'].numpy()
            ax1.quiver(0, 0, 0, original[0], original[1], original[2], color='blue', label=f'a{j+1}')
            
            # Plot previous orthonormal vectors
            for i in range(j):
                q_i = steps[i]['normalized'].numpy()
                ax1.quiver(0, 0, 0, q_i[0], q_i[1], q_i[2], color='green', label=f'q{i+1}')
            
            ax1.set_title(f"Step {j+1}: Original Vector")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.set_zlabel("z")
            ax1.set_xlim([-1.5, 1.5])
            ax1.set_ylim([-1.5, 1.5])
            ax1.set_zlim([-1.5, 1.5])
            
            # Plot projections
            ax2 = fig.add_subplot(132, projection='3d')
            
            # Plot original vector
            ax2.quiver(0, 0, 0, original[0], original[1], original[2], color='blue', alpha=0.5, label=f'a{j+1}')
            
            # Plot projections
            for proj in step['projections']:
                projection = proj['vector'].numpy()
                q_i = steps[proj['i']]['normalized'].numpy()
                
                ax2.quiver(0, 0, 0, q_i[0], q_i[1], q_i[2], color='green', alpha=0.7, label=f'q{proj["i"]+1}')
                ax2.quiver(0, 0, 0, projection[0], projection[1], projection[2], color='red', alpha=0.7, label=f'proj_{{{proj["i"]+1}}}(a{j+1})')
            
            # Draw orthogonalized vector
            orthogonalized = step['orthogonalized'].numpy()
            ax2.quiver(0, 0, 0, orthogonalized[0], orthogonalized[1], orthogonalized[2], color='purple', label=f'u{j+1}')
            
            ax2.set_title(f"Step {j+1}: Projections and Orthogonalization")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("z")
            ax2.set_xlim([-1.5, 1.5])
            ax2.set_ylim([-1.5, 1.5])
            ax2.set_zlim([-1.5, 1.5])
            
            # Plot normalized vector
            ax3 = fig.add_subplot(133, projection='3d')
            
            # Plot final orthonormal vector
            normalized = step['normalized'].numpy()
            ax3.quiver(0, 0, 0, normalized[0], normalized[1], normalized[2], color='purple', label=f'q{j+1}')
            
            # Plot previous orthonormal vectors
            for i in range(j):
                q_i = steps[i]['normalized'].numpy()
                ax3.quiver(0, 0, 0, q_i[0], q_i[1], q_i[2], color='green', alpha=0.7, label=f'q{i+1}')
            
            ax3.set_title(f"Step {j+1}: Normalized Vector")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_zlabel("z")
            ax3.set_xlim([-1.5, 1.5])
            ax3.set_ylim([-1.5, 1.5])
            ax3.set_zlim([-1.5, 1.5])
            
            plt.tight_layout()
            plt.show()
    
    # Print numerical details of each step
    for j, step in enumerate(steps):
        print(f"\nStep {j+1}: Processing column {j+1}")
        print(f"Original vector a{j+1}: {step['original_vector'].numpy()}")
        
        for proj in step['projections']:
            print(f"  Projection onto q{proj['i']+1}: coefficient = {proj['coefficient']:.4f}")
            print(f"  Projection vector: {proj['vector'].numpy()}")
        
        print(f"Orthogonalized vector u{j+1}: {step['orthogonalized'].numpy()}")
        print(f"Normalized vector q{j+1}: {step['normalized'].numpy()}")

# Create a small 2D matrix for visualization
def demonstrate_classical_gram_schmidt():
    """Demonstrate classical Gram-Schmidt algorithm in detail."""
    # Create a small 2D matrix
    A_2d = torch.tensor([
        [3.0, 1.0],
        [1.0, 2.0]
    ])
    
    # Apply classical Gram-Schmidt
    Q_2d, R_2d, steps_2d = classical_gram_schmidt_detailed(A_2d)
    
    # Display the matrices
    print("Original Matrix A:")
    print(A_2d.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q_2d.numpy())
    print("\nUpper Triangular Matrix R:")
    print(R_2d.numpy())
    
    # Visualize the process in 2D
    visualize_gram_schmidt_steps(A_2d, steps_2d, dim=2)
    
    # Create a small 3D matrix
    A_3d = torch.tensor([
        [3.0, 1.0, 2.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 3.0]
    ])
    
    # Apply classical Gram-Schmidt
    Q_3d, R_3d, steps_3d = classical_gram_schmidt_detailed(A_3d)
    
    # Display the matrices
    print("\nOriginal Matrix A (3D):")
    print(A_3d.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q_3d.numpy())
    print("\nUpper Triangular Matrix R:")
    print(R_3d.numpy())
    
    # Visualize the process in 3D
    visualize_gram_schmidt_steps(A_3d, steps_3d, dim=3)
    
    return A_2d, Q_2d, R_2d, A_3d, Q_3d, R_3d

# Demonstrate classical Gram-Schmidt algorithm
A_2d, Q_2d, R_2d, A_3d, Q_3d, R_3d = demonstrate_classical_gram_schmidt()

# %% [markdown]
# ### 1.1 Algorithm Analysis
# 
# Let's analyze the classical Gram-Schmidt algorithm in more detail.
# 
# **Computation**:
# - For each column $j$, we need to orthogonalize against all previous columns $i < j$
# - This requires $j$ dot products and vector subtractions
# - Overall computational complexity: $O(mn^2)$ for an $m \times n$ matrix
# 
# **Memory**:
# - We need to store the original matrix $A$, the orthogonal matrix $Q$, and the upper triangular matrix $R$
# - Memory complexity: $O(mn + n^2)$
# 
# **Numerical Stability**:
# - Classical Gram-Schmidt can suffer from numerical instability due to accumulation of rounding errors
# - This happens especially when the columns of $A$ are nearly linearly dependent

# %%
def analyze_gram_schmidt_complexity():
    """Analyze computational complexity of Gram-Schmidt."""
    # Define sizes to test
    sizes = range(100, 1001, 100)
    
    # Store timing results
    times = []
    
    for n in sizes:
        # Create a square matrix
        A = torch.rand(n, n, dtype=torch.float64)
        
        # Time classical Gram-Schmidt
        start_time = time.time()
        Q, R = classical_gram_schmidt(A)
        elapsed = time.time() - start_time
        
        times.append(elapsed)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-')
    plt.title("Computational Time vs. Matrix Size")
    plt.xlabel("Matrix Size (n×n)")
    plt.ylabel("Time (seconds)")
    plt.grid(True, alpha=0.3)
    
    # Plot theoretical complexity (scaled)
    theory = [s**3 * 1e-8 for s in sizes]  # O(n³) scaled to match
    plt.plot(sizes, theory, '--', label='O(n³)')
    
    plt.legend()
    plt.show()
    
    print(f"For a {sizes[-1]}×{sizes[-1]} matrix, classical Gram-Schmidt took {times[-1]:.4f} seconds")

# Simplified version of classical Gram-Schmidt without detailed steps
def classical_gram_schmidt(A):
    """Classical Gram-Schmidt algorithm."""
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
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

# Skip the full complexity analysis for brevity
print("Computational complexity of Gram-Schmidt is O(mn²) for an m×n matrix")
print("This is dominated by the dot product operations for each column")

# %% [markdown]
# ## 2. Modified Gram-Schmidt Algorithm
# 
# The modified Gram-Schmidt algorithm is a numerically more stable variant of the classical Gram-Schmidt process. Instead of orthogonalizing against all previous vectors at once, it orthogonalizes against each vector one at a time.
# 
# Here's the key difference:
# - **Classical GS**: Compute all projections from the original vector
# - **Modified GS**: Update the vector after each projection

# %%
def modified_gram_schmidt_detailed(A):
    """
    Implement modified Gram-Schmidt with detailed step-by-step visualization.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
        steps: Detailed steps for visualization
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    Q = torch.zeros((m, n), dtype=A.dtype)
    R = torch.zeros((n, n), dtype=A.dtype)
    
    # Initialize U as a copy of A
    U = A.clone()
    
    # Store intermediate steps for visualization
    steps = []
    
    for i in range(n):
        # Store the original vector and intermediate vectors
        step_data = {
            'i': i,
            'original_vector': U[:, i].clone(),
            'intermediate_vectors': [],
            'projections': []
        }
        
        # Compute the norm of the i-th column of U
        R[i, i] = torch.norm(U[:, i])
        
        # Normalize to get an orthonormal vector
        if R[i, i] > 1e-10:  # Check for numerical stability
            Q[:, i] = U[:, i] / R[i, i]
        else:
            Q[:, i] = torch.zeros(m, dtype=A.dtype)
        
        # Store the normalized vector
        step_data['normalized'] = Q[:, i].clone()
        
        # Orthogonalize remaining columns with respect to the i-th column of Q
        for j in range(i+1, n):
            # Record the vector before orthogonalization
            step_data['intermediate_vectors'].append({
                'j': j, 
                'before': U[:, j].clone()
            })
            
            # Calculate projection coefficient
            R[i, j] = torch.dot(Q[:, i], U[:, j])
            
            # Calculate projection vector
            projection = R[i, j] * Q[:, i]
            
            # Store projection for visualization
            step_data['projections'].append({
                'j': j,
                'coefficient': R[i, j].item(),
                'vector': projection.clone()
            })
            
            # Subtract projection
            U[:, j] = U[:, j] - projection
            
            # Record the vector after orthogonalization
            step_data['intermediate_vectors'][-1]['after'] = U[:, j].clone()
        
        # Add step to list
        steps.append(step_data)
    
    return Q, R, steps

def visualize_modified_gs_steps(A, steps, dim=2):
    """Visualize the Modified Gram-Schmidt process step by step."""
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    
    m, n = A.shape
    
    if dim > m:
        print(f"Warning: Can only visualize in {m} dimensions, not {dim}")
        dim = m
    
    if dim == 2:
        # For each step (processing one column)
        for i, step in enumerate(steps):
            # Create a figure for this step
            plt.figure(figsize=(15, 5))
            
            # Plot original column vector
            plt.subplot(1, 3, 1)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            original = step['original_vector'].numpy()
            plt.arrow(0, 0, original[0], original[1], head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue', label=f'u{i+1}')
            
            # Plot previous orthonormal vectors
            for j in range(i):
                q_j = steps[j]['normalized'].numpy()
                plt.arrow(0, 0, q_j[0], q_j[1], head_width=0.1, head_length=0.1, 
                         fc='green', ec='green', label=f'q{j+1}')
            
            plt.title(f"Step {i+1}: Original Vector u{i+1}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.legend()
            
            # Plot normalization
            plt.subplot(1, 3, 2)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            normalized = step['normalized'].numpy()
            
            # Draw original and normalized vector
            plt.arrow(0, 0, original[0], original[1], head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue', alpha=0.5, label=f'u{i+1}')
            plt.arrow(0, 0, normalized[0], normalized[1], head_width=0.1, head_length=0.1, 
                     fc='purple', ec='purple', label=f'q{i+1}')
            
            plt.title(f"Step {i+1}: Normalized Vector q{i+1}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.legend()
            
            # Plot effect on remaining vectors
            plt.subplot(1, 3, 3)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            
            # Draw the orthonormal vector
            plt.arrow(0, 0, normalized[0], normalized[1], head_width=0.1, head_length=0.1, 
                     fc='purple', ec='purple', label=f'q{i+1}')
            
            # Draw the effect on remaining vectors
            for idx, inter in enumerate(step['intermediate_vectors']):
                j = inter['j']
                before = inter['before'].numpy()
                after = inter['after'].numpy()
                proj = step['projections'][idx]['vector'].numpy()
                
                plt.arrow(0, 0, before[0], before[1], head_width=0.1, head_length=0.1, 
                         fc='blue', ec='blue', alpha=0.5, label=f'u{j+1} before')
                plt.arrow(0, 0, after[0], after[1], head_width=0.1, head_length=0.1, 
                         fc='green', ec='green', alpha=0.7, label=f'u{j+1} after')
                plt.arrow(0, 0, proj[0], proj[1], head_width=0.1, head_length=0.1, 
                         fc='red', ec='red', alpha=0.7, linestyle='dashed', 
                         label=f'proj_{{{i+1}}}(u{j+1})')
            
            plt.title(f"Step {i+1}: Effect on Remaining Vectors")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(-2, 2)
            plt.ylim(-2, 2)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
    
    # Print numerical details of each step
    for i, step in enumerate(steps):
        print(f"\nStep {i+1}: Processing column {i+1}")
        print(f"Original vector u{i+1}: {step['original_vector'].numpy()}")
        print(f"Normalized vector q{i+1}: {step['normalized'].numpy()}")
        
        for idx, proj in enumerate(step['projections']):
            j = proj['j']
            print(f"  Projection onto q{i+1} from u{j+1}: coefficient = {proj['coefficient']:.4f}")
            print(f"  Vector u{j+1} before: {step['intermediate_vectors'][idx]['before'].numpy()}")
            print(f"  Vector u{j+1} after: {step['intermediate_vectors'][idx]['after'].numpy()}")

def demonstrate_modified_gram_schmidt():
    """Demonstrate modified Gram-Schmidt algorithm in detail."""
    # Create a small 2D matrix
    A_2d = torch.tensor([
        [3.0, 1.0],
        [1.0, 2.0]
    ])
    
    # Apply modified Gram-Schmidt
    Q_2d, R_2d, steps_2d = modified_gram_schmidt_detailed(A_2d)
    
    # Display the matrices
    print("Original Matrix A:")
    print(A_2d.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q_2d.numpy())
    print("\nUpper Triangular Matrix R:")
    print(R_2d.numpy())
    
    # Visualize the process in 2D
    visualize_modified_gs_steps(A_2d, steps_2d, dim=2)
    
    return A_2d, Q_2d, R_2d, steps_2d

# Demonstration of modified Gram-Schmidt
A_2d_mgs, Q_2d_mgs, R_2d_mgs, steps_2d_mgs = demonstrate_modified_gram_schmidt()

# %% [markdown]
# ### 2.1 Comparing Classical and Modified Gram-Schmidt
# 
# Let's compare the numerical stability of the classical and modified Gram-Schmidt algorithms, especially for ill-conditioned matrices:

# %%
def compare_gs_stability():
    """Compare the numerical stability of classical and modified Gram-Schmidt."""
    # Create an ill-conditioned matrix
    n = 5
    
    # Start with orthogonal columns
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    
    # Create a diagonal matrix with a large condition number
    kappa = 1e10  # Condition number
    S = torch.diag(torch.tensor([1.0] + [1.0/np.sqrt(kappa)] * (n-1)))
    
    # Combine to get an ill-conditioned matrix
    A = Q @ S
    
    # Apply classical Gram-Schmidt
    Q_classical, R_classical = classical_gram_schmidt(A)
    
    # Apply modified Gram-Schmidt
    def modified_gram_schmidt(A):
        if isinstance(A, np.ndarray):
            A = torch.tensor(A, dtype=torch.float64)
        
        m, n = A.shape
        Q = torch.zeros((m, n), dtype=A.dtype)
        R = torch.zeros((n, n), dtype=A.dtype)
        
        # Initialize U as a copy of A
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
    
    Q_modified, R_modified = modified_gram_schmidt(A)
    
    # Also compute QR with PyTorch's built-in function
    Q_torch, R_torch = torch.linalg.qr(A)
    
    # Measure orthogonality
    QTQ_classical = Q_classical.T @ Q_classical
    QTQ_modified = Q_modified.T @ Q_modified
    QTQ_torch = Q_torch.T @ Q_torch
    
    # Measure reconstruction error
    rec_error_classical = torch.norm(A - Q_classical @ R_classical).item()
    rec_error_modified = torch.norm(A - Q_modified @ R_modified).item()
    rec_error_torch = torch.norm(A - Q_torch @ R_torch).item()
    
    # Measure orthogonality error
    ortho_error_classical = torch.norm(QTQ_classical - torch.eye(n)).item()
    ortho_error_modified = torch.norm(QTQ_modified - torch.eye(n)).item()
    ortho_error_torch = torch.norm(QTQ_torch - torch.eye(n)).item()
    
    # Display the orthogonality matrices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(QTQ_classical.numpy(), annot=True, fmt=".2e", cmap=blue_cmap)
    plt.title(f"Q^T Q - Classical GS\nError: {ortho_error_classical:.2e}")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(QTQ_modified.numpy(), annot=True, fmt=".2e", cmap=blue_cmap)
    plt.title(f"Q^T Q - Modified GS\nError: {ortho_error_modified:.2e}")
    
    plt.subplot(1, 3, 3)
    sns.heatmap(QTQ_torch.numpy(), annot=True, fmt=".2e", cmap=blue_cmap)
    plt.title(f"Q^T Q - PyTorch QR\nError: {ortho_error_torch:.2e}")
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Matrix Condition Number: {kappa:.1e}")
    print("\nOrthogonality Error (||Q^T Q - I||):")
    print(f"Classical Gram-Schmidt: {ortho_error_classical:.2e}")
    print(f"Modified Gram-Schmidt: {ortho_error_modified:.2e}")
    print(f"PyTorch QR: {ortho_error_torch:.2e}")
    
    print("\nReconstruction Error (||A - QR||):")
    print(f"Classical Gram-Schmidt: {rec_error_classical:.2e}")
    print(f"Modified Gram-Schmidt: {rec_error_modified:.2e}")
    print(f"PyTorch QR: {rec_error_torch:.2e}")
    
    # Vary condition number and observe orthogonality error
    condition_numbers = [1e4, 1e6, 1e8, 1e10, 1e12]
    classical_errors = []
    modified_errors = []
    torch_errors = []
    
    for kappa in condition_numbers:
        # Create matrix with specified condition number
        S = torch.diag(torch.tensor([1.0] + [1.0/np.sqrt(kappa)] * (n-1)))
        A = Q @ S
        
        # Apply methods
        Q_classical, R_classical = classical_gram_schmidt(A)
        Q_modified, R_modified = modified_gram_schmidt(A)
        Q_torch, R_torch = torch.linalg.qr(A)
        
        # Measure orthogonality error
        ortho_error_classical = torch.norm(Q_classical.T @ Q_classical - torch.eye(n)).item()
        ortho_error_modified = torch.norm(Q_modified.T @ Q_modified - torch.eye(n)).item()
        ortho_error_torch = torch.norm(Q_torch.T @ Q_torch - torch.eye(n)).item()
        
        classical_errors.append(ortho_error_classical)
        modified_errors.append(ortho_error_modified)
        torch_errors.append(ortho_error_torch)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.loglog(condition_numbers, classical_errors, 'o-', label='Classical GS')
    plt.loglog(condition_numbers, modified_errors, 's-', label='Modified GS')
    plt.loglog(condition_numbers, torch_errors, '^-', label='PyTorch QR')
    
    plt.xlabel("Condition Number")
    plt.ylabel("Orthogonality Error")
    plt.title("Effect of Condition Number on Orthogonality")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return classical_errors, modified_errors, torch_errors

# Compare stability of GS methods
classical_errors, modified_errors, torch_errors = compare_gs_stability()

# %% [markdown]
# ## 3. Householder Reflections
# 
# Householder reflections provide another approach to QR decomposition that offers excellent numerical stability. A Householder reflection is a transformation that reflects a vector about a hyperplane.
# 
# The key idea is to iteratively zero out elements below the diagonal, one column at a time:
# 1. For each column, create a reflection that maps the column to a multiple of the first unit vector
# 2. Apply this reflection to the entire matrix
# 3. The product of these reflections gives $Q$

# %%
def householder_detailed(A):
    """
    Implement QR decomposition using Householder reflections with detailed steps.
    
    Args:
        A: Input matrix as a PyTorch tensor
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
        steps: Detailed steps for visualization
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    R = A.clone()
    Q = torch.eye(m, dtype=A.dtype)
    
    # Store steps for visualization
    steps = []
    
    for k in range(min(m-1, n)):
        # Extract the column we want to transform
        x = R[k:, k]
        
        # Store the current state before reflection
        step_data = {
            'k': k,
            'column_before': x.clone(),
            'R_before': R.clone(),
            'Q_before': Q.clone()
        }
        
        # Construct the Householder vector
        e1 = torch.zeros_like(x)
        e1[0] = 1.0
        
        # Compute norm of x
        alpha = torch.norm(x)
        # Ensure proper sign of alpha to avoid cancellation
        if x[0] < 0:
            alpha = -alpha
        
        # Construct the Householder vector
        u = x - alpha * e1
        v = u / torch.norm(u)  # Normalize
        
        # Store Householder vector
        step_data['householder_vector'] = v.clone()
        
        # Apply the Householder reflection to R
        R[k:, k:] = R[k:, k:] - 2.0 * torch.outer(v, torch.matmul(v, R[k:, k:]))
        
        # Apply the Householder reflection to Q
        Q[:, k:] = Q[:, k:] - 2.0 * torch.matmul(Q[:, k:], torch.outer(v, v))
        
        # Store the state after reflection
        step_data['column_after'] = R[k:, k].clone()
        step_data['R_after'] = R.clone()
        step_data['Q_after'] = Q.clone()
        
        steps.append(step_data)
    
    # Final Q and R
    Q = Q.T
    
    # Ensure the diagonal of R is positive
    for i in range(min(m, n)):
        if R[i, i] < 0:
            R[i, i:] = -R[i, i:]
            Q[:, i] = -Q[:, i]
    
    return Q, R, steps

def visualize_householder_steps(A, steps):
    """Visualize the Householder QR decomposition process."""
    if isinstance(A, torch.Tensor):
        A = A.numpy()
    
    m, n = A.shape
    
    # For each step
    for k, step in enumerate(steps):
        plt.figure(figsize=(15, 10))
        
        # Plot the R matrix before reflection
        plt.subplot(2, 3, 1)
        sns.heatmap(step['R_before'].numpy(), annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title(f"Step {k+1}: R Before Reflection")
        
        # Plot the column before reflection
        plt.subplot(2, 3, 2)
        column_before = np.zeros(m)
        column_before[step['k']:] = step['column_before'].numpy()
        sns.heatmap(column_before.reshape(-1, 1), annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title(f"Column {step['k']+1} Before")
        
        # Plot the Householder vector
        plt.subplot(2, 3, 3)
        v_full = np.zeros(m - step['k'])
        v_full = step['householder_vector'].numpy()
        v_viz = np.zeros(m)
        v_viz[step['k']:] = v_full
        sns.heatmap(v_viz.reshape(-1, 1), annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title("Householder Vector v")
        
        # Plot the R matrix after reflection
        plt.subplot(2, 3, 4)
        sns.heatmap(step['R_after'].numpy(), annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title(f"Step {k+1}: R After Reflection")
        
        # Plot the column after reflection
        plt.subplot(2, 3, 5)
        column_after = np.zeros(m)
        column_after[step['k']:] = step['column_after'].numpy()
        sns.heatmap(column_after.reshape(-1, 1), annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title(f"Column {step['k']+1} After")
        
        # Plot the current Q matrix 
        plt.subplot(2, 3, 6)
        sns.heatmap(step['Q_after'].T.numpy(), annot=True, fmt=".2f", cmap=blue_cmap)
        plt.title(f"Current Q Matrix")
        
        plt.tight_layout()
        plt.show()
        
        # For 2D or 3D matrices, add a geometric visualization
        if m == 2 or m == 3:
            # Geometric visualization in 2D
            if m == 2:
                plt.figure(figsize=(12, 6))
                
                # Get the original column and Householder vector
                col = step['column_before'].numpy()
                v = step['householder_vector'].numpy()
                
                # Compute the reflection plane (line in 2D)
                # The reflection plane is perpendicular to v
                # In 2D, we can represent it with a line through the origin
                if v[1] != 0:
                    slope = -v[0] / v[1]
                    x = np.linspace(-3, 3, 100)
                    y = slope * x
                else:
                    x = np.zeros(100)
                    y = np.linspace(-3, 3, 100)
                
                plt.subplot(1, 2, 1)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
                # Draw the original column vector
                plt.arrow(0, 0, col[0], col[1], head_width=0.1, head_length=0.1, 
                         fc='blue', ec='blue', label='Original Column')
                
                # Draw the Householder vector
                plt.arrow(0, 0, v[0], v[1], head_width=0.1, head_length=0.1, 
                         fc='red', ec='red', label='Householder Vector')
                
                # Draw the reflection plane
                plt.plot(x, y, 'g--', label='Reflection Plane')
                
                plt.axis('equal')
                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.grid(True)
                plt.title("Householder Vector and Reflection Plane")
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                
                # Draw the original column vector
                plt.arrow(0, 0, col[0], col[1], head_width=0.1, head_length=0.1, 
                         fc='blue', ec='blue', alpha=0.5, label='Original Column')
                
                # Draw the reflected column
                col_after = step['column_after'].numpy()
                plt.arrow(0, 0, col_after[0], col_after[1], head_width=0.1, head_length=0.1, 
                         fc='green', ec='green', label='Reflected Column')
                
                # Draw the reflection plane
                plt.plot(x, y, 'g--', label='Reflection Plane')
                
                plt.axis('equal')
                plt.xlim(-3, 3)
                plt.ylim(-3, 3)
                plt.grid(True)
                plt.title("Effect of Householder Reflection")
                plt.legend()
                
                plt.tight_layout()
                plt.show()
            
            # For 3D visualization (not implemented for brevity)
            elif m == 3:
                pass
    
    # Print numerical details of the process
    for k, step in enumerate(steps):
        print(f"\nStep {k+1}: Zeroing below diagonal in column {step['k']+1}")
        print(f"Column before reflection: {step['column_before'].numpy()}")
        print(f"Householder vector: {step['householder_vector'].numpy()}")
        print(f"Column after reflection: {step['column_after'].numpy()}")

def demonstrate_householder():
    """Demonstrate Householder reflections for QR decomposition."""
    # Create a small 2D matrix for geometric visualization
    A_2d = torch.tensor([
        [3.0, 1.0],
        [1.0, 2.0]
    ])
    
    # Apply Householder QR
    Q_2d, R_2d, steps_2d = householder_detailed(A_2d)
    
    # Display the matrices
    print("Original Matrix A:")
    print(A_2d.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q_2d.numpy())
    print("\nUpper Triangular Matrix R:")
    print(R_2d.numpy())
    
    # Visualize the process
    visualize_householder_steps(A_2d, steps_2d)
    
    # Create a larger matrix
    A_larger = torch.tensor([
        [4.0, 1.0, 3.0],
        [2.0, 5.0, 2.0],
        [1.0, 0.0, 6.0],
        [3.0, 2.0, 1.0]
    ])
    
    # Apply Householder QR
    Q_larger, R_larger, steps_larger = householder_detailed(A_larger)
    
    # Display the matrices
    print("\nOriginal Matrix A (4×3):")
    print(A_larger.numpy())
    print("\nOrthogonal Matrix Q:")
    print(Q_larger.numpy())
    print("\nUpper Triangular Matrix R:")
    print(R_larger.numpy())
    
    # Visualize the process
    visualize_householder_steps(A_larger, steps_larger)
    
    return A_2d, Q_2d, R_2d, A_larger, Q_larger, R_larger

# Demonstrate Householder reflections
A_2d_hh, Q_2d_hh, R_2d_hh, A_larger_hh, Q_larger_hh, R_larger_hh = demonstrate_householder()

# %% [markdown]
# ## 4. Givens Rotations
# 
# Givens rotations provide an alternative approach to QR decomposition. Instead of using reflections, they use rotations to zero out elements one at a time. This approach is especially useful for sparse matrices, as it can preserve sparsity better than Householder reflections.
# 
# A Givens rotation rotates a plane by a specific angle to zero out one element:

# %%
def givens_rotation(A):
    """
    Implement QR decomposition using Givens rotations.
    
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
    
    # For each column
    for j in range(n):
        # For each element below the diagonal in this column
        for i in range(m-1, j, -1):
            # Skip if the element is already zero
            if abs(R[i, j]) < 1e-10:
                continue
            
            # Compute the Givens rotation parameters
            a = R[i-1, j]
            b = R[i, j]
            r = torch.sqrt(a*a + b*b)
            c = a / r  # cosine
            s = -b / r  # sine
            
            # Create the Givens rotation matrix (identity with a 2x2 rotation)
            G = torch.eye(m, dtype=A.dtype)
            G[i-1, i-1] = c
            G[i-1, i] = s
            G[i, i-1] = -s
            G[i, i] = c
            
            # Apply the rotation to R
            R = G @ R
            
            # Update Q (we use G^T since we're building Q from the left)
            Q = Q @ G.T
    
    return Q, R

def demonstrate_givens():
    """Demonstrate Givens rotations for QR decomposition."""
    # Create a small test matrix
    A = torch.tensor([
        [3.0, 1.0, 2.0],
        [1.0, 2.0, 0.0],
        [2.0, 0.0, 3.0]
    ])
    
    # Apply Givens rotations
    Q_givens, R_givens = givens_rotation(A)
    
    # For comparison, use Householder reflections
    Q_house, R_house = householder_detailed(A)[0:2]
    
    # Display the matrices
    print("Original Matrix A:")
    print(A.numpy())
    print("\nOrthogonal Matrix Q (Givens):")
    print(Q_givens.numpy())
    print("\nUpper Triangular Matrix R (Givens):")
    print(R_givens.numpy())
    
    # Verify the decomposition
    rec_error_givens = torch.norm(A - Q_givens @ R_givens).item()
    ortho_error_givens = torch.norm(Q_givens.T @ Q_givens - torch.eye(3)).item()
    
    rec_error_house = torch.norm(A - Q_house @ R_house).item()
    ortho_error_house = torch.norm(Q_house.T @ Q_house - torch.eye(3)).item()
    
    print("\nGivens Rotations:")
    print(f"Reconstruction error: {rec_error_givens:.2e}")
    print(f"Orthogonality error: {ortho_error_givens:.2e}")
    
    print("\nHouseholder Reflections:")
    print(f"Reconstruction error: {rec_error_house:.2e}")
    print(f"Orthogonality error: {ortho_error_house:.2e}")
    
    # Create a larger sparse matrix
    n = 10
    sparse_A = torch.zeros((n, n), dtype=torch.float64)
    
    # Add some non-zero elements (sparse pattern)
    for i in range(n):
        sparse_A[i, i] = i + 1  # Diagonal
        if i < n-1:
            sparse_A[i, i+1] = 0.5  # Superdiagonal
        if i > 0:
            sparse_A[i, i-1] = 0.3  # Subdiagonal
    
    # Apply both methods
    Q_givens_sparse, R_givens_sparse = givens_rotation(sparse_A)
    Q_house_sparse, R_house_sparse, _ = householder_detailed(sparse_A)
    
    # Measure sparsity
    def count_zeros(matrix, threshold=1e-10):
        return torch.sum(torch.abs(matrix) < threshold).item()
    
    zeros_givens = count_zeros(R_givens_sparse)
    zeros_house = count_zeros(R_house_sparse)
    
    print("\nSparsity Comparison (number of zero elements in R):")
    print(f"Givens Rotations: {zeros_givens} zeros out of {n*n} elements")
    print(f"Householder Reflections: {zeros_house} zeros out of {n*n} elements")
    
    # Visualize the sparsity patterns
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.spy(sparse_A.numpy(), markersize=5)
    plt.title("Original Sparse Matrix")
    
    plt.subplot(1, 3, 2)
    plt.spy(R_givens_sparse.numpy(), markersize=5)
    plt.title("R from Givens Rotations")
    
    plt.subplot(1, 3, 3)
    plt.spy(R_house_sparse.numpy(), markersize=5)
    plt.title("R from Householder Reflections")
    
    plt.tight_layout()
    plt.show()
    
    return A, Q_givens, R_givens, sparse_A, R_givens_sparse, R_house_sparse

# Demonstrate Givens rotations
A_givens, Q_givens, R_givens, sparse_A, R_givens_sparse, R_house_sparse = demonstrate_givens()

# %% [markdown]
# ## 5. Performance Comparison of QR Algorithms

# %%
def compare_qr_performance():
    """Compare the performance of different QR decomposition algorithms."""
    # Define matrix sizes to test
    sizes = range(20, 201, 20)
    
    # Store timing results
    times_classical = []
    times_modified = []
    times_householder = []
    times_givens = []
    times_builtin = []
    
    # Store accuracy results
    ortho_classical = []
    ortho_modified = []
    ortho_householder = []
    ortho_givens = []
    ortho_builtin = []
    
    for n in sizes:
        # Create a random matrix
        A = torch.rand(n, n, dtype=torch.float64)
        
        # Classical Gram-Schmidt
        start_time = time.time()
        Q, R = classical_gram_schmidt(A)
        times_classical.append(time.time() - start_time)
        ortho_classical.append(torch.norm(Q.T @ Q - torch.eye(n)).item())
        
        # Modified Gram-Schmidt
        def modified_gram_schmidt(A):
            """Modified Gram-Schmidt algorithm."""
            if isinstance(A, np.ndarray):
                A = torch.tensor(A, dtype=torch.float64)
            
            m, n = A.shape
            Q = torch.zeros((m, n), dtype=A.dtype)
            R = torch.zeros((n, n), dtype=A.dtype)
            
            # Initialize U as a copy of A
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
        
        start_time = time.time()
        Q, R = modified_gram_schmidt(A)
        times_modified.append(time.time() - start_time)
        ortho_modified.append(torch.norm(Q.T @ Q - torch.eye(n)).item())
        
        # Householder reflections
        def householder(A):
            """Householder reflections for QR decomposition."""
            if isinstance(A, np.ndarray):
                A = torch.tensor(A, dtype=torch.float64)
            
            m, n = A.shape
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
            
            return Q, R
        
        start_time = time.time()
        Q, R = householder(A)
        times_householder.append(time.time() - start_time)
        ortho_householder.append(torch.norm(Q.T @ Q - torch.eye(n)).item())
        
        # Givens rotations
        start_time = time.time()
        Q, R = givens_rotation(A)
        times_givens.append(time.time() - start_time)
        ortho_givens.append(torch.norm(Q.T @ Q - torch.eye(n)).item())
        
        # Built-in QR
        start_time = time.time()
        Q, R = torch.linalg.qr(A)
        times_builtin.append(time.time() - start_time)
        ortho_builtin.append(torch.norm(Q.T @ Q - torch.eye(n)).item())
    
    # Plot the results
    plt.figure(figsize=(12, 10))
    
    # Timing comparison
    plt.subplot(2, 1, 1)
    plt.plot(sizes, times_classical, 'o-', label='Classical Gram-Schmidt')
    plt.plot(sizes, times_modified, 's-', label='Modified Gram-Schmidt')
    plt.plot(sizes, times_householder, '^-', label='Householder Reflections')
    plt.plot(sizes, times_givens, 'x-', label='Givens Rotations')
    plt.plot(sizes, times_builtin, 'd-', label='PyTorch QR')
    
    plt.xlabel("Matrix Size (n×n)")
    plt.ylabel("Time (seconds)")
    plt.title("QR Decomposition Time vs. Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Orthogonality comparison
    plt.subplot(2, 1, 2)
    plt.semilogy(sizes, ortho_classical, 'o-', label='Classical Gram-Schmidt')
    plt.semilogy(sizes, ortho_modified, 's-', label='Modified Gram-Schmidt')
    plt.semilogy(sizes, ortho_householder, '^-', label='Householder Reflections')
    plt.semilogy(sizes, ortho_givens, 'x-', label='Givens Rotations')
    plt.semilogy(sizes, ortho_builtin, 'd-', label='PyTorch QR')
    
    plt.xlabel("Matrix Size (n×n)")
    plt.ylabel("Orthogonality Error")
    plt.title("QR Decomposition Accuracy vs. Matrix Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary for largest size
    largest_size = sizes[-1]
    print(f"Performance Comparison for {largest_size}×{largest_size} Matrix:")
    print("-" * 80)
    print(f"{'Method':<25} {'Time (s)':<15} {'Orthogonality Error':<20}")
    print("-" * 80)
    
    methods = ["Classical Gram-Schmidt", "Modified Gram-Schmidt", 
               "Householder Reflections", "Givens Rotations", "PyTorch QR"]
    times = [times_classical[-1], times_modified[-1], 
             times_householder[-1], times_givens[-1], times_builtin[-1]]
    errors = [ortho_classical[-1], ortho_modified[-1], 
              ortho_householder[-1], ortho_givens[-1], ortho_builtin[-1]]
    
    for method, t, error in zip(methods, times, errors):
        print(f"{method:<25} {t:<15.6f} {error:<20.2e}")
    
    # Return the data for further analysis
    return {
        'sizes': sizes,
        'times': {
            'classical': times_classical,
            'modified': times_modified,
            'householder': times_householder,
            'givens': times_givens,
            'builtin': times_builtin
        },
        'orthogonality': {
            'classical': ortho_classical,
            'modified': ortho_modified,
            'householder': ortho_householder,
            'givens': ortho_givens,
            'builtin': ortho_builtin
        }
    }

# Skip the full performance comparison for brevity
# Instead, provide a summary
print("Performance Summary of QR Algorithms:")
print("1. Classical Gram-Schmidt: Simple to implement but numerically unstable")
print("2. Modified Gram-Schmidt: Improved stability with similar computation cost")
print("3. Householder Reflections: Excellent stability, efficient for dense matrices")
print("4. Givens Rotations: Good stability, efficient for sparse matrices")
print("5. Built-in QR: Optimized implementation, usually fastest and most stable")

# %% [markdown]
# ## 6. Block QR Algorithms
# 
# For large matrices, block algorithms can improve performance by leveraging cache locality and optimized matrix-matrix operations. Let's implement a simple block QR algorithm:

# %%
def block_qr(A, block_size=32):
    """
    Implement block QR decomposition using Householder reflections.
    
    Args:
        A: Input matrix as a PyTorch tensor
        block_size: Size of the blocks
        
    Returns:
        Q: Orthogonal matrix
        R: Upper triangular matrix
    """
    if isinstance(A, np.ndarray):
        A = torch.tensor(A, dtype=torch.float64)
    
    m, n = A.shape
    R = A.clone()
    Q = torch.eye(m, dtype=A.dtype)
    
    # Process the matrix in block columns
    for j in range(0, n, block_size):
        # Adjust block size for the last block
        current_block_size = min(block_size, n - j)
        
        # Work on the current block column
        for k in range(j, min(j + current_block_size, m-1)):
            # Extract the column we want to transform
            x = R[k:, k]
            
            # Skip if the column is already zeroed below the diagonal
            if torch.norm(x[1:]) < 1e-10:
                continue
            
            # Construct the Householder vector
            e1 = torch.zeros_like(x)
            e1[0] = 1.0
            
            alpha = torch.norm(x)
            if x[0] < 0:
                alpha = -alpha
                
            u = x - alpha * e1
            v = u / torch.norm(u)
            
            # Apply the Householder reflection to the remaining part of R
            R[k:, k:] = R[k:, k:] - 2.0 * torch.outer(v, torch.matmul(v, R[k:, k:]))
            
            # Update Q
            Q[:, k:] = Q[:, k:] - 2.0 * torch.matmul(Q[:, k:], torch.outer(v, v))
    
    # Transpose Q to get the final orthogonal matrix
    Q = Q.T
    
    return Q, R

def demonstrate_block_qr():
    """Demonstrate block QR decomposition."""
    # Create a larger matrix for testing
    n = 100
    A = torch.rand(n, n, dtype=torch.float64)
    
    # Compare block QR with standard QR
    block_sizes = [5, 10, 20, 50]
    times_block = []
    times_standard = []
    
    for bs in block_sizes:
        # Block QR
        start_time = time.time()
        Q_block, R_block = block_qr(A, block_size=bs)
        block_time = time.time() - start_time
        times_block.append(block_time)
        
        # Standard QR (just once)
        if len(times_standard) == 0:
            start_time = time.time()
            Q_standard, R_standard = torch.linalg.qr(A)
            standard_time = time.time() - start_time
            times_standard = [standard_time] * len(block_sizes)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(block_sizes, times_block, 'o-', label='Block QR')
    plt.axhline(y=times_standard[0], color='r', linestyle='--', label='Standard QR')
    
    plt.xlabel("Block Size")
    plt.ylabel("Time (seconds)")
    plt.title(f"Block QR Performance vs. Block Size (n={n})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add annotations
    for i, bs in enumerate(block_sizes):
        plt.annotate(f"{times_block[i]:.4f}s", 
                    (bs, times_block[i]), 
                    textcoords="offset points",
                    xytext=(0, 10), 
                    ha='center')
    
    plt.annotate(f"{times_standard[0]:.4f}s", 
                (block_sizes[-1], times_standard[0]), 
                textcoords="offset points",
                xytext=(30, 0), 
                ha='center')
    
    plt.show()
    
    # Verify the decomposition
    error_block = torch.norm(A - Q_block @ R_block).item()
    ortho_error_block = torch.norm(Q_block.T @ Q_block - torch.eye(n)).item()
    
    error_standard = torch.norm(A - Q_standard @ R_standard).item()
    ortho_error_standard = torch.norm(Q_standard.T @ Q_standard - torch.eye(n)).item()
    
    print(f"Block QR (block size {block_sizes[-1]}):")
    print(f"Reconstruction error: {error_block:.2e}")
    print(f"Orthogonality error: {ortho_error_block:.2e}")
    
    print(f"\nStandard QR:")
    print(f"Reconstruction error: {error_standard:.2e}")
    print(f"Orthogonality error: {ortho_error_standard:.2e}")
    
    return A, Q_block, R_block, Q_standard, R_standard

# Skip the block QR demonstration for brevity
print("Block QR Algorithms:")
print("- Process the matrix in blocks to improve cache locality")
print("- Can leverage optimized BLAS routines for matrix-matrix operations")
print("- Performance depends on the block size and hardware architecture")
print("- Optimal block size typically matches the cache size of the processor")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored different algorithms for computing QR decomposition:
# 
# 1. **Classical Gram-Schmidt**: Simple to implement but suffers from numerical instability, especially for ill-conditioned matrices.
# 
# 2. **Modified Gram-Schmidt**: Improves numerical stability by updating the vectors after each projection.
# 
# 3. **Householder Reflections**: Offers excellent numerical stability and efficiency for dense matrices.
# 
# 4. **Givens Rotations**: Particularly useful for sparse matrices as they can preserve sparsity patterns.
# 
# 5. **Block Algorithms**: Improve performance for large matrices by leveraging cache locality.
# 
# Key takeaways:
# 
# - For general-purpose use, Householder reflections provide the best balance of stability and efficiency.
# - For sparse matrices, Givens rotations can be more efficient.
# - For very large matrices, block algorithms can provide performance improvements.
# - In practice, optimized libraries typically use a combination of these techniques.
# 
# The QR decomposition is a fundamental tool in numerical linear algebra with numerous applications, which we'll explore in the next notebook.

# %%