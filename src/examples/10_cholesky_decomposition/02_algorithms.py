# %% [markdown]
# # Cholesky Decomposition: Algorithms and Variants
# 
# In this notebook, we explore various algorithms and variants of the Cholesky decomposition, including:
# 
# 1. Block Cholesky decomposition
# 2. Modified Cholesky decomposition
# 3. Incomplete Cholesky factorization
# 4. Handling positive semi-definite matrices
# 5. Rank-1 updates to Cholesky factors
# 
# These variations are useful in different contexts and applications, such as optimization, solving linear systems efficiently, and handling ill-conditioned matrices.

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
def create_positive_definite_matrix(n, method="random"):
    """Create a positive definite matrix of size n×n."""
    if method == "random":
        # Create a random matrix
        B = torch.randn(n, n)
        # A = B*B^T is guaranteed to be positive definite (if B is full rank)
        A = B @ B.T
        # Add a small value to the diagonal to ensure positive definiteness
        A = A + torch.eye(n) * 1e-5
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
# ## 1. Block Cholesky Decomposition
# 
# Block Cholesky decomposition is a variation of the standard algorithm that operates on blocks of the matrix rather than individual elements. This approach can be more efficient for large matrices, especially when combined with parallel computing.
# 
# For a matrix $A$ partitioned into blocks:
# 
# $$A = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}$$
# 
# The block Cholesky decomposition gives:
# 
# $$L = \begin{bmatrix} L_{11} & 0 \\ L_{21} & L_{22} \end{bmatrix}$$
# 
# Where:
# - $L_{11}$ is the Cholesky factor of $A_{11}$
# - $L_{21} = A_{21} L_{11}^{-T}$
# - $L_{22}$ is the Cholesky factor of $A_{22} - L_{21}L_{21}^T$

# %%
def block_cholesky(A, block_size=2):
    """
    Compute the Cholesky decomposition using a block algorithm.
    
    Parameters:
        A (torch.Tensor): Positive definite matrix
        block_size (int): Size of blocks to process
        
    Returns:
        L (torch.Tensor): Lower triangular matrix
    """
    n = A.shape[0]
    L = torch.zeros_like(A)
    
    # Process matrix in blocks
    for i in range(0, n, block_size):
        # Determine the current block size (might be smaller at the end)
        current_block_size = min(block_size, n - i)
        
        # Extract blocks
        A11 = A[i:i+current_block_size, i:i+current_block_size]
        
        # Compute L11 block (Cholesky of A11)
        L11 = torch.linalg.cholesky(A11)
        L[i:i+current_block_size, i:i+current_block_size] = L11
        
        # For the remaining rows in this block column
        if i + current_block_size < n:
            # Extract A21 block
            A21 = A[i+current_block_size:, i:i+current_block_size]
            
            # Compute L21 block
            L21 = A21 @ torch.inverse(L11.T)
            L[i+current_block_size:, i:i+current_block_size] = L21
            
            # Update the remaining part of A for next iteration
            A[i+current_block_size:, i+current_block_size:] -= L21 @ L21.T
    
    return L

# Create a 6×6 matrix to test block Cholesky
A = create_positive_definite_matrix(6, method="random")

# Standard Cholesky decomposition
L_standard = torch.linalg.cholesky(A)

# Block Cholesky decomposition
L_block = block_cholesky(A.clone(), block_size=2)

# Verification
print("Error between standard and block Cholesky:", torch.norm(L_standard - L_block).item())

# %% [markdown]
# We can visualize the block processing with a larger matrix:

# %%
def visualize_block_cholesky(A, block_size=2):
    """Visualize the block Cholesky process."""
    n = A.shape[0]
    A_copy = A.clone()  # Work on a copy to leave original intact
    L = torch.zeros_like(A_copy)
    
    plt.figure(figsize=(15, 5 * ((n // block_size) + 1)))
    
    # Plot original matrix
    plt.subplot(n // block_size + 2, 2, 1)
    sns.heatmap(A_copy.numpy(), annot=True, fmt=".1f", cmap="Blues", linewidths=.5)
    plt.title("Original Matrix A")
    
    step = 2
    
    # Process matrix in blocks
    for i in range(0, n, block_size):
        # Determine the current block size (might be smaller at the end)
        current_block_size = min(block_size, n - i)
        
        # Extract blocks
        A11 = A_copy[i:i+current_block_size, i:i+current_block_size]
        
        # Compute L11 block (Cholesky of A11)
        L11 = torch.linalg.cholesky(A11)
        L[i:i+current_block_size, i:i+current_block_size] = L11
        
        # For the remaining rows in this block column
        if i + current_block_size < n:
            # Extract A21 block
            A21 = A_copy[i+current_block_size:, i:i+current_block_size]
            
            # Compute L21 block
            L21 = A21 @ torch.inverse(L11.T)
            L[i+current_block_size:, i:i+current_block_size] = L21
            
            # Update the remaining part of A for next iteration
            A_copy[i+current_block_size:, i+current_block_size:] -= L21 @ L21.T
        
        # Plot the current state
        plt.subplot(n // block_size + 2, 2, step)
        
        # Create a mask for the blocks we've processed
        processed_mask = torch.zeros_like(A, dtype=bool)
        processed_mask[i+current_block_size:, i:i+current_block_size] = True  # L21 block
        processed_mask[i:i+current_block_size, i:i+current_block_size] = True  # L11 block
        
        sns.heatmap(L.numpy(), annot=True, fmt=".1f", cmap="Blues", linewidths=.5, 
                   mask=~processed_mask.numpy() & (L == 0).numpy())
        plt.title(f"Step {i//block_size + 1}: Processed blocks L11, L21")
        
        # Plot the updated matrix A
        plt.subplot(n // block_size + 2, 2, step + 1)
        
        # Create a mask for the updated part of A
        updated_mask = torch.zeros_like(A, dtype=bool)
        if i + current_block_size < n:
            updated_mask[i+current_block_size:, i+current_block_size:] = True  # Updated part
        
        sns.heatmap(A_copy.numpy(), annot=True, fmt=".1f", cmap="Oranges", linewidths=.5, 
                   mask=~updated_mask.numpy())
        plt.title(f"Step {i//block_size + 1}: Updated remaining submatrix")
        
        step += 2
    
    # Final result
    plt.subplot(n // block_size + 2, 2, step)
    sns.heatmap(L.numpy(), annot=True, fmt=".1f", cmap="Blues", linewidths=.5)
    plt.title("Final Cholesky Factor L")
    
    plt.tight_layout()
    plt.show()
    
    return L

# Create a 6×6 matrix to visualize block Cholesky
A = create_positive_definite_matrix(6, method="random")
L_block_viz = visualize_block_cholesky(A, block_size=2)

# %% [markdown]
# ## 2. Modified Cholesky Decomposition
# 
# Modified Cholesky decomposition is useful when dealing with matrices that are close to positive definite but may not be numerically positive definite due to roundoff errors or other numerical issues.
# 
# The idea is to compute a decomposition $LL^T$ of a matrix $A + E$, where $E$ is a "small" perturbation matrix that ensures positive definiteness.

# %%
def modified_cholesky(A, delta=1e-6):
    """
    Compute a modified Cholesky decomposition, ensuring positive definiteness.
    
    Parameters:
        A (torch.Tensor): Symmetric matrix that may not be positive definite
        delta (float): Minimum allowed value for diagonal elements
        
    Returns:
        L (torch.Tensor): Lower triangular matrix
        E (torch.Tensor): Perturbation added to make A positive definite
    """
    n = A.shape[0]
    # Make a copy to avoid modifying the input
    A_mod = A.clone()
    E = torch.zeros_like(A)
    
    # Initialize L as zeros
    L = torch.zeros_like(A)
    
    for j in range(n):
        # Process diagonal element
        if j > 0:
            # Update diagonal element based on previously computed columns
            for k in range(j):
                A_mod[j, j] -= L[j, k]**2
        
        # If diagonal element is too small, modify it
        if A_mod[j, j] < delta:
            correction = delta - A_mod[j, j]
            A_mod[j, j] = delta
            E[j, j] = correction
        
        # Set diagonal element of L
        L[j, j] = torch.sqrt(A_mod[j, j])
        
        # Process off-diagonal elements
        for i in range(j+1, n):
            if j > 0:
                # Update based on previously computed columns
                for k in range(j):
                    A_mod[i, j] -= L[i, k] * L[j, k]
            
            # Set off-diagonal element of L
            L[i, j] = A_mod[i, j] / L[j, j]
    
    return L, E

# Create a matrix that is not quite positive definite
def create_nearly_positive_definite_matrix(n, eigenvalue_min=-0.1):
    """Create a symmetric matrix with eigenvalues that may be slightly negative."""
    # Start with a random symmetric matrix
    A = torch.randn(n, n)
    A = 0.5 * (A + A.T)
    
    # Get its eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    
    # Modify the eigenvalues to have some small negative values
    modified_eigenvalues = eigenvalues.clone()
    modified_eigenvalues[-2:] = eigenvalue_min  # Set the smallest eigenvalues to a small negative value
    
    # Reconstruct the matrix
    A_modified = eigenvectors @ torch.diag(modified_eigenvalues) @ eigenvectors.T
    
    return A_modified

# Create a nearly positive definite matrix
A_nearly_pd = create_nearly_positive_definite_matrix(5)

# Check eigenvalues
eigenvalues = torch.linalg.eigvalsh(A_nearly_pd)
print("Eigenvalues of nearly positive definite matrix:", eigenvalues)
print("Is A positive definite?", torch.all(eigenvalues > 0).item())

# Standard Cholesky would fail
try:
    L_standard = torch.linalg.cholesky(A_nearly_pd)
    print("Standard Cholesky succeeded (unexpected)")
except Exception as e:
    print("Standard Cholesky failed with error:", str(e))

# Modified Cholesky
L_modified, E = modified_cholesky(A_nearly_pd)
print("\nPerturbation matrix E:")
print(E)

# Verify the modified factorization
A_perturbed = A_nearly_pd + E
reconstructed_A = L_modified @ L_modified.T
print("\nError in modified Cholesky reconstruction:", torch.norm(A_perturbed - reconstructed_A).item())

# Check eigenvalues of perturbed matrix
eigenvalues_perturbed = torch.linalg.eigvalsh(A_perturbed)
print("Eigenvalues of perturbed matrix:", eigenvalues_perturbed)
print("Is perturbed matrix positive definite?", torch.all(eigenvalues_perturbed > 0).item())

# %% [markdown]
# Let's visualize the original matrix, the perturbation, and the modified matrix:

# %%
def visualize_modified_cholesky(A, L, E):
    """Visualize the modified Cholesky decomposition."""
    plt.figure(figsize=(18, 6))
    
    # Original matrix
    plt.subplot(1, 4, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title("Original Matrix A")
    
    # Perturbation matrix
    plt.subplot(1, 4, 2)
    sns.heatmap(E.numpy(), annot=True, fmt=".2f", cmap="Reds", linewidths=.5)
    plt.title("Perturbation Matrix E")
    
    # Perturbed matrix
    plt.subplot(1, 4, 3)
    sns.heatmap((A + E).numpy(), annot=True, fmt=".2f", cmap="Greens", linewidths=.5)
    plt.title("Perturbed Matrix A + E")
    
    # Cholesky factor of the perturbed matrix
    plt.subplot(1, 4, 4)
    sns.heatmap(L.numpy(), annot=True, fmt=".2f", cmap="Purples", linewidths=.5)
    plt.title("Cholesky Factor L")
    
    plt.tight_layout()
    plt.show()

# Visualize the modified Cholesky decomposition
visualize_modified_cholesky(A_nearly_pd, L_modified, E)

# %% [markdown]
# ## 3. Incomplete Cholesky Factorization
# 
# Incomplete Cholesky factorization is a sparse approximation of the full Cholesky decomposition. It's useful for preconditioning large sparse systems.
# 
# The key idea is to only calculate entries of $L$ that correspond to non-zero entries in the sparsity pattern of the original matrix $A$.

# %%
def incomplete_cholesky(A, drop_tol=1e-4):
    """
    Compute an incomplete Cholesky factorization with drop tolerance.
    
    Parameters:
        A (torch.Tensor): Sparse positive definite matrix
        drop_tol (float): Tolerance for dropping small entries
        
    Returns:
        L (torch.Tensor): Sparse lower triangular matrix
    """
    n = A.shape[0]
    # Make a copy to avoid modifying the input
    A_copy = A.clone()
    
    # Initialize L as zeros
    L = torch.zeros_like(A)
    
    for k in range(n):
        # Diagonal element
        if A_copy[k, k] <= 0:
            # If diagonal is not positive, add a small value
            A_copy[k, k] = 1e-8
        
        L[k, k] = torch.sqrt(A_copy[k, k])
        
        # Update the current column of L
        for i in range(k+1, n):
            if abs(A_copy[i, k]) > drop_tol:
                L[i, k] = A_copy[i, k] / L[k, k]
            else:
                L[i, k] = 0  # Drop small entries
        
        # Update the trailing submatrix
        for i in range(k+1, n):
            for j in range(k+1, i+1):  # Only update lower triangular part
                if L[i, k] != 0 and L[j, k] != 0:  # Only update if both entries are non-zero
                    A_copy[i, j] -= L[i, k] * L[j, k]
                    A_copy[j, i] = A_copy[i, j]  # Keep symmetric
    
    return L

# Create a sparse positive definite matrix
def create_sparse_positive_definite_matrix(n, density=0.3):
    """Create a sparse positive definite matrix."""
    # Create a random sparse matrix
    indices = torch.randint(0, n, (2, int(n * n * density)))
    values = torch.randn(indices.shape[1])
    sparse = torch.sparse_coo_tensor(indices, values, (n, n)).to_dense()
    
    # Make it symmetric
    sparse = sparse + sparse.T
    
    # Add a diagonal shift to ensure positive definiteness
    sparse = sparse + torch.eye(n) * (n + 1)
    
    # Zero out some entries to make it sparse
    mask = torch.rand(n, n) < density
    sparse = sparse * mask
    
    # Ensure symmetry
    sparse = 0.5 * (sparse + sparse.T)
    
    return sparse

# Create a sparse positive definite matrix
A_sparse = create_sparse_positive_definite_matrix(8, density=0.4)
print("Sparsity pattern (1 = non-zero):")
print((A_sparse != 0).int())

# Compute incomplete Cholesky factorization
L_incomplete = incomplete_cholesky(A_sparse, drop_tol=1e-2)

# Compare with full Cholesky
L_full = torch.linalg.cholesky(A_sparse)

# Compute reconstructed matrices
A_approx = L_incomplete @ L_incomplete.T
A_exact = L_full @ L_full.T

print("\nError in incomplete Cholesky reconstruction:", torch.norm(A_sparse - A_approx).item())
print("Error in full Cholesky reconstruction:", torch.norm(A_sparse - A_exact).item())

# Calculate sparsity levels
print(f"\nSparsity of original matrix: {torch.sum(A_sparse != 0).item()}/{A_sparse.numel()} elements")
print(f"Sparsity of incomplete Cholesky: {torch.sum(L_incomplete != 0).item()}/{L_incomplete.numel()} elements")
print(f"Sparsity of full Cholesky: {torch.sum(L_full != 0).item()}/{L_full.numel()} elements")

# %% [markdown]
# Let's visualize the sparsity patterns:

# %%
def visualize_sparsity(A, L_incomplete, L_full):
    """Visualize the sparsity patterns."""
    plt.figure(figsize=(15, 5))
    
    # Original matrix
    plt.subplot(1, 3, 1)
    plt.spy(A.numpy(), markersize=10)
    plt.title("Original Matrix A")
    
    # Incomplete Cholesky
    plt.subplot(1, 3, 2)
    plt.spy(L_incomplete.numpy(), markersize=10)
    plt.title("Incomplete Cholesky Factor")
    
    # Full Cholesky
    plt.subplot(1, 3, 3)
    plt.spy(L_full.numpy(), markersize=10)
    plt.title("Full Cholesky Factor")
    
    plt.tight_layout()
    plt.show()

# Visualize sparsity patterns
visualize_sparsity(A_sparse, L_incomplete, L_full)

# %% [markdown]
# ## 4. Handling Positive Semi-Definite Matrices
# 
# A positive semi-definite matrix has eigenvalues that are non-negative (≥ 0), but some may be zero.
# Standard Cholesky decomposition requires strictly positive eigenvalues. For semi-definite matrices,
# we need alternative approaches.

# %%
def create_positive_semidefinite_matrix(n, rank_deficiency=1):
    """Create a positive semidefinite matrix with specified rank deficiency."""
    # Create a random matrix with specific rank
    effective_rank = n - rank_deficiency
    X = torch.randn(n, effective_rank)
    A = X @ X.T
    
    # Ensure it's symmetric (it should be already, but for numerical stability)
    A = 0.5 * (A + A.T)
    
    return A

# Create a positive semidefinite matrix
A_semidefinite = create_positive_semidefinite_matrix(5, rank_deficiency=1)

# Check eigenvalues
eigenvalues = torch.linalg.eigvalsh(A_semidefinite)
print("Eigenvalues of semi-definite matrix:", eigenvalues)
print("Rank:", torch.sum(eigenvalues > 1e-10).item())

# Standard Cholesky might fail depending on how close eigenvalues are to zero
try:
    L_standard = torch.linalg.cholesky(A_semidefinite)
    print("Standard Cholesky succeeded, but may be numerically unstable")
except Exception as e:
    print("Standard Cholesky failed with error:", str(e))

# %% [markdown]
# ### Pivoted Cholesky Decomposition
# 
# Pivoted Cholesky is useful for rank-deficient matrices. It rearranges rows and columns to ensure numerical stability.

# %%
def pivoted_cholesky(A, tol=1e-10):
    """
    Compute a pivoted Cholesky decomposition.
    
    Parameters:
        A (torch.Tensor): Positive semi-definite matrix
        tol (float): Tolerance for detecting rank deficiency
        
    Returns:
        L (torch.Tensor): Lower triangular matrix
        P (torch.Tensor): Permutation matrix
        rank (int): Numerical rank of A
    """
    n = A.shape[0]
    A_copy = A.clone()
    
    # Initialize L as zeros
    L = torch.zeros_like(A)
    
    # Initialize permutation as identity
    perm = torch.arange(n)
    
    # Compute the factorization with pivoting
    rank = 0
    for k in range(n):
        # Find the maximum diagonal element in the remaining submatrix
        diag_vals = torch.diag(A_copy)[k:]
        max_idx = torch.argmax(diag_vals) + k
        
        # Break if the maximum diagonal element is numerically zero
        if A_copy[max_idx, max_idx] < tol:
            break
            
        # Swap rows and columns if needed
        if max_idx != k:
            # Swap rows and columns in A
            A_copy[[k, max_idx], :] = A_copy[[max_idx, k], :]
            A_copy[:, [k, max_idx]] = A_copy[:, [max_idx, k]]
            
            # Swap rows in L (up to column k-1)
            L[[k, max_idx], :k] = L[[max_idx, k], :k]
            
            # Update permutation
            perm[k], perm[max_idx] = perm[max_idx], perm[k]
        
        # Compute diagonal element
        L[k, k] = torch.sqrt(A_copy[k, k])
        
        # Compute column k of L
        for i in range(k+1, n):
            L[i, k] = A_copy[i, k] / L[k, k]
        
        # Update remaining submatrix
        for i in range(k+1, n):
            for j in range(k+1, n):
                A_copy[i, j] -= L[i, k] * L[j, k]
        
        rank += 1
    
    # Create permutation matrix from permutation vector
    P = torch.zeros(n, n)
    for i in range(n):
        P[i, perm[i]] = 1
    
    return L, P, rank

# Apply pivoted Cholesky to our semidefinite matrix
L_pivoted, P, numerical_rank = pivoted_cholesky(A_semidefinite)

print(f"Numerical rank detected: {numerical_rank}")
print("\nPermutation matrix P:")
print(P)

# Verify the factorization: P^T A P ≈ L L^T
reconstructed_A = L_pivoted @ L_pivoted.T
permuted_A = P.T @ A_semidefinite @ P

print("\nError in pivoted Cholesky reconstruction:", torch.norm(permuted_A - reconstructed_A).item())

# %% [markdown]
# Let's visualize the pivoted Cholesky decomposition:

# %%
def visualize_pivoted_cholesky(A, L, P, rank):
    """Visualize the pivoted Cholesky decomposition."""
    plt.figure(figsize=(15, 5))
    
    # Original matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title("Original Matrix A")
    
    # Permuted matrix
    plt.subplot(1, 3, 2)
    permuted_A = P.T @ A @ P
    sns.heatmap(permuted_A.numpy(), annot=True, fmt=".2f", cmap="Greens", linewidths=.5)
    plt.title("Permuted Matrix P^T A P")
    
    # Cholesky factor
    plt.subplot(1, 3, 3)
    masked_L = L.clone()
    # Mask the part below the numerical rank
    if rank < L.shape[0]:
        for i in range(rank, L.shape[0]):
            masked_L[i, rank:] = 0
            
    sns.heatmap(masked_L.numpy(), annot=True, fmt=".2f", cmap="Purples", linewidths=.5)
    plt.title(f"Cholesky Factor L (rank = {rank})")
    
    plt.tight_layout()
    plt.show()

# Visualize the pivoted Cholesky decomposition
visualize_pivoted_cholesky(A_semidefinite, L_pivoted, P, numerical_rank)

# %% [markdown]
# ## 5. Rank-1 Updates to Cholesky Factors
# 
# In many applications, such as Kalman filtering or sequential least squares, we need to update a Cholesky factorization when the underlying matrix receives a small update.
# 
# For a rank-1 update $A + vv^T$, we can efficiently update the Cholesky factor without recomputing it from scratch.

# %%
def cholesky_update(L, v):
    """
    Update a Cholesky factorization for A + v*v^T.
    
    Parameters:
        L (torch.Tensor): Cholesky factor of A
        v (torch.Tensor): Update vector
        
    Returns:
        L_new (torch.Tensor): Updated Cholesky factor
    """
    n = L.shape[0]
    L_new = L.clone()
    
    # Copy v for the update
    w = v.clone()
    
    for k in range(n):
        # Apply rotations to maintain triangular structure
        r = torch.sqrt(L_new[k, k]**2 + w[k]**2)
        c = L_new[k, k] / r
        s = w[k] / r
        
        # Update diagonal element
        L_new[k, k] = r
        
        # Update remaining elements in this column
        for i in range(k+1, n):
            # Apply rotation to L_new[i, k] and w[i]
            L_new[i, k], w[i] = c * L_new[i, k] + s * w[i], -s * L_new[i, k] + c * w[i]
    
    return L_new

# Create a positive definite matrix and its Cholesky factor
A = create_positive_definite_matrix(4)
L = torch.linalg.cholesky(A)

# Create a random vector for the update
v = torch.randn(4)

# Update the matrix directly
A_updated = A + torch.outer(v, v)

# Compute Cholesky of updated matrix from scratch
L_updated_direct = torch.linalg.cholesky(A_updated)

# Update the Cholesky factor incrementally
L_updated_incremental = cholesky_update(L, v)

print("Error between direct and incremental updates:", 
      torch.norm(L_updated_direct - L_updated_incremental).item())

# %% [markdown]
# Let's visualize the update process:

# %%
def visualize_cholesky_update(A, L, v, L_updated):
    """Visualize the Cholesky update process."""
    plt.figure(figsize=(15, 5))
    
    # Original matrix
    plt.subplot(1, 4, 1)
    sns.heatmap(A.numpy(), annot=True, fmt=".2f", cmap="Blues", linewidths=.5)
    plt.title("Original Matrix A")
    
    # Update matrix
    plt.subplot(1, 4, 2)
    update = torch.outer(v, v)
    sns.heatmap(update.numpy(), annot=True, fmt=".2f", cmap="Reds", linewidths=.5)
    plt.title("Update v*v^T")
    
    # Updated matrix
    plt.subplot(1, 4, 3)
    A_updated = A + update
    sns.heatmap(A_updated.numpy(), annot=True, fmt=".2f", cmap="Greens", linewidths=.5)
    plt.title("Updated Matrix A + v*v^T")
    
    # Updated Cholesky factor
    plt.subplot(1, 4, 4)
    sns.heatmap(L_updated.numpy(), annot=True, fmt=".2f", cmap="Purples", linewidths=.5)
    plt.title("Updated Cholesky Factor")
    
    plt.tight_layout()
    plt.show()

# Visualize the Cholesky update
visualize_cholesky_update(A, L, v, L_updated_incremental)

# %% [markdown]
# ### Performance Comparison: Update vs. Recompute
# 
# Let's compare the performance of updating the Cholesky factor versus recomputing it from scratch:

# %%
# Create a larger matrix for timing comparison
n = 500
A = create_positive_definite_matrix(n)
L = torch.linalg.cholesky(A)

# Create random vectors for updates
num_updates = 10
update_times = []
recompute_times = []

for i in range(num_updates):
    v = torch.randn(n)
    
    # Time the incremental update
    start_time = time.time()
    L_updated_incremental = cholesky_update(L, v)
    update_time = time.time() - start_time
    update_times.append(update_time)
    
    # Update the matrix directly
    A = A + torch.outer(v, v)
    
    # Time the direct recomputation
    start_time = time.time()
    L_updated_direct = torch.linalg.cholesky(A)
    recompute_time = time.time() - start_time
    recompute_times.append(recompute_time)
    
    # Use the incremental update for next iteration
    L = L_updated_incremental
    
    print(f"Update {i+1}:")
    print(f"  Update time: {update_time:.6f} seconds")
    print(f"  Recompute time: {recompute_time:.6f} seconds")
    print(f"  Speedup: {recompute_time/update_time:.2f}x")

# Plot timing comparison
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_updates+1), update_times, 'o-', label='Incremental Update')
plt.plot(range(1, num_updates+1), recompute_times, 's-', label='Full Recomputation')
plt.xlabel('Update Number')
plt.ylabel('Computation Time (seconds)')
plt.title('Cholesky Update vs. Recomputation Performance')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Summary
# 
# In this notebook, we explored various algorithms and variants of Cholesky decomposition:
# 
# 1. **Block Cholesky decomposition**: processes matrices in blocks for improved efficiency
# 2. **Modified Cholesky decomposition**: handles matrices that are not quite positive definite
# 3. **Incomplete Cholesky factorization**: creates sparse approximations for large sparse systems
# 4. **Pivoted Cholesky**: deals with rank-deficient positive semi-definite matrices
# 5. **Rank-1 Updates**: efficiently updates Cholesky factors without full recomputation
# 
# Each of these variants is valuable in different contexts, such as numerical optimization, solving linear systems efficiently, and handling ill-conditioned or sparse matrices.
# 
# In the next notebook, we'll explore practical applications of Cholesky decomposition.