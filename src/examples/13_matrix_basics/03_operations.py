# %% [markdown]
# # Matrix Basics: Operations
# 
# This notebook explores basic matrix operations in PyTorch. We'll cover:
# 
# - Element-wise operations (addition, subtraction, multiplication, division)
# - Matrix transposition 
# - Matrix concatenation and stacking
# - In-place operations and memory efficiency
# 
# These fundamental operations serve as building blocks for more complex linear algebra algorithms.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set default figure size
plt.rcParams["figure.figsize"] = (10, 8)

# %% [markdown]
# ## Element-wise Operations
# 
# Element-wise operations are performed on corresponding elements of tensors. Let's explore the basic arithmetic operations: addition, subtraction, multiplication, and division.

# %%
# Create two matrices
A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
B = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# %% [markdown]
# ### Element-wise Addition

# %%
# Element-wise addition
C = A + B  # Equivalent to torch.add(A, B)
print("A + B:")
print(C)

# %% [markdown]
# ### Element-wise Subtraction

# %%
# Element-wise subtraction
D = A - B  # Equivalent to torch.sub(A, B)
print("A - B:")
print(D)

# %% [markdown]
# ### Element-wise Multiplication

# %%
# Element-wise multiplication (Hadamard product)
E = A * B  # Equivalent to torch.mul(A, B)
print("A * B (element-wise):")
print(E)

# %% [markdown]
# ### Element-wise Division

# %%
# Element-wise division
F = A / B  # Equivalent to torch.div(A, B)
print("A / B:")
print(F)

# %% [markdown]
# Let's visualize these element-wise operations to better understand them:

# %%
def visualize_element_wise_operation(A, B, result, operation):
    """Visualize element-wise operation between two matrices."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Matrix A
    sns.heatmap(A.numpy(), annot=True, fmt=".1f", cmap="Blues", 
               ax=axes[0], cbar=False)
    axes[0].set_title("Matrix A")
    
    # Matrix B
    sns.heatmap(B.numpy(), annot=True, fmt=".1f", cmap="Oranges", 
               ax=axes[1], cbar=False)
    axes[1].set_title("Matrix B")
    
    # Result matrix
    sns.heatmap(result.numpy(), annot=True, fmt=".2f", cmap="Greens", 
               ax=axes[2], cbar=False)
    axes[2].set_title(f"A {operation} B")
    
    plt.tight_layout()
    plt.show()

# Visualize element-wise operations
visualize_element_wise_operation(A, B, C, "+")
visualize_element_wise_operation(A, B, D, "-")
visualize_element_wise_operation(A, B, E, "*")
visualize_element_wise_operation(A, B, F, "/")

# %% [markdown]
# ### Other Element-wise Operations

# %%
# Element-wise power
G = A ** 2  # Equivalent to torch.pow(A, 2)
print("A ** 2:")
print(G)

# Element-wise square root
H = torch.sqrt(A)
print("\nSqrt(A):")
print(H)

# Element-wise exponential
I = torch.exp(A)
print("\nExp(A):")
print(I)

# Element-wise logarithm
J = torch.log(A)
print("\nLog(A):")
print(J)

# Element-wise absolute value
K = torch.abs(torch.tensor([[-1, 2, -3], [4, -5, 6]]))
print("\nAbs([-1, 2, -3; 4, -5, 6]):")
print(K)

# %% [markdown]
# ## Operations with Scalars
# 
# Element-wise operations can also be performed between matrices and scalars:

# %%
# Create a matrix
A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("Matrix A:")
print(A)

# Scalar operations
scalar = 2.0

# Addition with scalar
add_scalar = A + scalar
print("\nA + 2.0:")
print(add_scalar)

# Multiplication with scalar
mul_scalar = A * scalar
print("\nA * 2.0:")
print(mul_scalar)

# Division by scalar
div_scalar = A / scalar
print("\nA / 2.0:")
print(div_scalar)

# %% [markdown]
# Let's visualize scalar operations:

# %%
def visualize_scalar_operation(A, scalar, result, operation):
    """Visualize operation between a matrix and a scalar."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matrix A
    sns.heatmap(A.numpy(), annot=True, fmt=".1f", cmap="Blues", 
               ax=axes[0], cbar=False)
    axes[0].set_title(f"Matrix A")
    
    # Result matrix
    sns.heatmap(result.numpy(), annot=True, fmt=".1f", cmap="Greens", 
               ax=axes[1], cbar=False)
    axes[1].set_title(f"A {operation} {scalar}")
    
    plt.tight_layout()
    plt.show()

# Visualize scalar operations
visualize_scalar_operation(A, scalar, add_scalar, "+")
visualize_scalar_operation(A, scalar, mul_scalar, "*")
visualize_scalar_operation(A, scalar, div_scalar, "/")

# %% [markdown]
# ## Matrix Transposition
# 
# The transpose of a matrix swaps its rows and columns. In PyTorch, this is done using the `.t()` method for 2D tensors or the `.transpose()` method for higher dimensions.

# %%
# Create a matrix
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Matrix A:")
print(A)
print(f"Shape: {A.shape}")

# Transpose
A_T = A.t()
print("\nTranspose of A:")
print(A_T)
print(f"Shape: {A_T.shape}")

# For higher dimensions, we use transpose with dimension indices
B = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\nTensor B:")
print(B)
print(f"Shape: {B.shape}")

# Transpose dimensions 0 and 2
B_T = B.transpose(0, 2)
print("\nB with dimensions 0 and 2 transposed:")
print(B_T)
print(f"Shape: {B_T.shape}")

# %% [markdown]
# Let's visualize matrix transposition:

# %%
def visualize_transpose(A, A_T):
    """Visualize matrix transposition."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original matrix
    sns.heatmap(A.numpy(), annot=True, fmt="d", cmap="Blues", 
               ax=axes[0], cbar=False)
    axes[0].set_title(f"Original Matrix\nShape: {A.shape}")
    
    # Transposed matrix
    sns.heatmap(A_T.numpy(), annot=True, fmt="d", cmap="Blues", 
               ax=axes[1], cbar=False)
    axes[1].set_title(f"Transposed Matrix\nShape: {A_T.shape}")
    
    plt.tight_layout()
    plt.show()

# Visualize transposition
visualize_transpose(A, A_T)

# %% [markdown]
# ## Matrix Concatenation and Stacking
# 
# PyTorch provides several functions to combine tensors in different ways:
# 
# - `torch.cat()`: Concatenates tensors along a specified dimension
# - `torch.stack()`: Adds a new dimension and stacks tensors along it

# %%
# Create two matrices
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
B = torch.tensor([[7, 8, 9], [10, 11, 12]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Concatenate along dimension 0 (rows)
C_row = torch.cat((A, B), dim=0)
print("\nConcatenated along rows (dim=0):")
print(C_row)
print(f"Shape: {C_row.shape}")

# Concatenate along dimension 1 (columns)
C_col = torch.cat((A, B), dim=1)
print("\nConcatenated along columns (dim=1):")
print(C_col)
print(f"Shape: {C_col.shape}")

# Stack matrices (adds new dimension)
D = torch.stack((A, B), dim=0)
print("\nStacked along new dimension (dim=0):")
print(D)
print(f"Shape: {D.shape}")

# Stack along different dimension
E = torch.stack((A, B), dim=2)
print("\nStacked along new dimension (dim=2):")
print(E)
print(f"Shape: {E.shape}")

# %% [markdown]
# Let's visualize concatenation and stacking operations:

# %%
def visualize_concatenation(A, B, result, dim, operation="Concatenation"):
    """Visualize tensor concatenation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Matrix A
    sns.heatmap(A.numpy(), annot=True, fmt="d", cmap="Blues", 
               ax=axes[0], cbar=False)
    axes[0].set_title(f"Matrix A\nShape: {A.shape}")
    
    # Matrix B
    sns.heatmap(B.numpy(), annot=True, fmt="d", cmap="Oranges", 
               ax=axes[1], cbar=False)
    axes[1].set_title(f"Matrix B\nShape: {B.shape}")
    
    # Result matrix
    if operation == "Stacking" and result.dim() > 2:
        # For stacked tensors, we show the first slice
        if dim == 0:
            sns.heatmap(result[0].numpy(), annot=True, fmt="d", cmap="Greens", 
                       ax=axes[2], cbar=False)
            axes[2].set_title(f"{operation} (dim={dim})\nFirst slice shown\nFull shape: {result.shape}")
        elif dim == 2:
            sns.heatmap(result[:,:,0].numpy(), annot=True, fmt="d", cmap="Greens", 
                       ax=axes[2], cbar=False)
            axes[2].set_title(f"{operation} (dim={dim})\nFirst slice shown\nFull shape: {result.shape}")
    else:
        sns.heatmap(result.numpy(), annot=True, fmt="d", cmap="Greens", 
                   ax=axes[2], cbar=False)
        axes[2].set_title(f"{operation} (dim={dim})\nShape: {result.shape}")
    
    plt.tight_layout()
    plt.show()

# Visualize concatenation
visualize_concatenation(A, B, C_row, 0)
visualize_concatenation(A, B, C_col, 1)

# Visualize stacking
visualize_concatenation(A, B, D, 0, "Stacking")
visualize_concatenation(A, B, E, 2, "Stacking")

# %% [markdown]
# ## In-place Operations
# 
# PyTorch supports in-place operations that modify tensors directly without creating a new tensor. These operations are memory-efficient but can interfere with autograd (automatic differentiation).

# %%
# Create a matrix
A = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("Original A:")
print(A)

# In-place addition
A.add_(5)  # Note the underscore suffix for in-place operations
print("\nAfter A.add_(5):")
print(A)

# In-place multiplication
A.mul_(2)
print("\nAfter A.mul_(2):")
print(A)

# Create a fresh tensor for more examples
B = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print("\nNew tensor B:")
print(B)

# In-place subtraction with another tensor
C = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
B.sub_(C)
print("\nAfter B.sub_(C):")
print(B)

# %% [markdown]
# ## Comparison of Out-of-place vs In-place Operations
# 
# Let's compare the performance and memory usage of in-place versus out-of-place operations:

# %%
import time

# Function to measure time and memory for operations
def measure_performance(op_name, operation, iterations=10000):
    start_time = time.time()
    for _ in range(iterations):
        result = operation()
    end_time = time.time()
    return (end_time - start_time) * 1000 / iterations  # ms per iteration

# Create large tensors for benchmarking
large_A = torch.randn(1000, 1000)
large_B = torch.randn(1000, 1000)

# Measure out-of-place addition
out_of_place_time = measure_performance(
    "Out-of-place addition",
    lambda: large_A + large_B
)

# Measure in-place addition
in_place_time = measure_performance(
    "In-place addition",
    lambda: large_A.clone().add_(large_B)  # Clone to avoid modifying original
)

print(f"Average time for out-of-place addition: {out_of_place_time:.4f} ms")
print(f"Average time for in-place addition: {in_place_time:.4f} ms")
print(f"Speed improvement with in-place: {(out_of_place_time - in_place_time) / out_of_place_time * 100:.2f}%")

# %% [markdown]
# Let's visualize this performance difference:

# %%
operation_types = ["Out-of-place", "In-place"]
times = [out_of_place_time, in_place_time]

plt.figure(figsize=(10, 6))
bars = plt.bar(operation_types, times, color=['skyblue', 'salmon'])
plt.title("Performance Comparison: Out-of-place vs In-place Addition")
plt.ylabel("Time per operation (ms)")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f} ms', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## When to Use In-place Operations
# 
# In-place operations should be used with caution:
# 
# - **Use when**: Memory is a concern and you're working with large tensors
# - **Avoid when**: Computing gradients (autograd) is needed for the tensor
# - **Avoid when**: You need to keep the original tensor values for later use

# %%
# Example showing potential issues with in-place operations and autograd

# Create tensors that require gradients
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# This works fine
z = x + y
z.backward(torch.ones_like(z))
print("Gradients after regular addition:")
print(f"x.grad: {x.grad}")
print(f"y.grad: {y.grad}")

# Reset gradients
x.grad.zero_()
y.grad.zero_()

# Try an in-place operation
try:
    x.add_(y)  # This will raise an error
    x.backward(torch.ones_like(x))
except RuntimeError as e:
    print(f"\nError with in-place operation: {e}")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored basic matrix operations in PyTorch:
# 
# - Element-wise operations (addition, subtraction, multiplication, division)
# - Operations with scalars
# - Matrix transposition
# - Concatenation and stacking of matrices
# - In-place operations and their performance benefits and limitations
# 
# These fundamental operations form the building blocks for more complex linear algebra operations and deep learning algorithms. Understanding these basics is crucial for effective tensor manipulation in PyTorch.