# %% [markdown]
# # Vector Spaces and Subspaces
# 
# In this notebook, we will explore the concept of vector spaces and subspaces, which are fundamental in linear algebra. We'll focus on building intuition through visualizations and practical examples using PyTorch.
# 
# ## What is a Vector Space?
# 
# A vector space is a collection of objects called vectors that can be added together and multiplied by scalars (numbers). These operations must satisfy certain axioms, such as associativity, commutativity of addition, and distributivity of scalar multiplication.
# 
# The most common vector space is $\mathbb{R}^n$, the space of all n-dimensional real vectors.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import seaborn as sns
sns.set_style("whitegrid")

# For better looking plots
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Visualizing Vectors in R²
# 
# Let's start by visualizing some vectors in a 2D space (R²).

# %%
def plot_vectors_2d(vectors, colors, labels=None):
    """Plot vectors in 2D."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Creating limits for the plot
    max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors]) * 1.2
    
    for i, v in enumerate(vectors):
        plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, 
                  head_length=max_val*0.08, fc=colors[i], ec=colors[i], label=labels[i] if labels else f"Vector {i+1}")
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vectors in R²')
    if labels:
        plt.legend()
    plt.show()

# Example vectors
v1 = torch.tensor([1.0, 2.0])
v2 = torch.tensor([3.0, 1.0])
v3 = v1 + v2  # Vector addition
v4 = 2 * v1   # Scalar multiplication

vectors = [v1, v2, v3, v4]
colors = ['blue', 'red', 'green', 'purple']
labels = ['v1', 'v2', 'v1 + v2', '2 * v1']

plot_vectors_2d(vectors, colors, labels)

# %% [markdown]
# The figure above illustrates two key properties of vector spaces:
# 
# 1. **Vector Addition**: The sum of vectors v1 and v2 (green arrow) follows the parallelogram rule.
# 2. **Scalar Multiplication**: Multiplying a vector v1 by scalar 2 (purple arrow) stretches the vector by a factor of 2 in the same direction.
# 
# ## Linear Combinations
# 
# A linear combination of vectors is formed by multiplying each vector by a scalar and adding the results. If v₁, v₂, ..., vₙ are vectors and c₁, c₂, ..., cₙ are scalars, then:
# 
# $c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + ... + c_n \mathbf{v}_n$
# 
# is a linear combination of these vectors.

# %%
def plot_linear_combinations_2d(v1, v2, coeffs_list, show_span=False):
    """Plot linear combinations of two vectors in 2D with coefficients."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Draw original vectors
    max_val = max(max(abs(v1[0]), abs(v1[1])), max(abs(v2[0]), abs(v2[1]))) * 3
    
    # Optional: Show the span of the vectors (the plane they generate)
    if show_span:
        grid_points = 10
        span_alpha = 0.1
        x = np.linspace(-max_val, max_val, grid_points)
        y = np.linspace(-max_val, max_val, grid_points)
        X, Y = np.meshgrid(x, y)
        
        # Calculate coefficients for spanning the plane
        # This is a simplified approach for 2D vectors
        det = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(det) > 1e-10:  # Check if vectors are linearly independent
            plt.fill_between([-max_val, max_val], -max_val, max_val, color='lightgray', alpha=span_alpha)
            plt.text(max_val*0.7, max_val*0.7, "Span(v1, v2)", fontsize=12, color='gray')
    
    # Draw the original vectors
    plt.arrow(0, 0, v1[0], v1[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='blue', ec='blue', label='v1')
    plt.arrow(0, 0, v2[0], v2[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='red', ec='red', label='v2')
    
    # Draw linear combinations
    for i, coeffs in enumerate(coeffs_list):
        a, b = coeffs
        linear_comb = a * v1 + b * v2
        plt.arrow(0, 0, linear_comb[0], linear_comb[1], head_width=max_val*0.05, 
                  head_length=max_val*0.08, fc='green', ec='green', 
                  label=f'{a}*v1 + {b}*v2' if i == 0 else "")
        
        # Add parallelogram for visualization
        plt.plot([0, a*v1[0], linear_comb[0], b*v2[0], 0], 
                 [0, a*v1[1], linear_comb[1], b*v2[1], 0], 
                 'k--', alpha=0.3)
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Combinations of Vectors in R²')
    plt.legend()
    plt.show()

# Using our previously defined vectors
coeffs_list = [(0.5, 1.5), (1.5, -0.5), (-1.0, 2.0)]
plot_linear_combinations_2d(v1, v2, coeffs_list, show_span=True)

# %% [markdown]
# ## Vector Subspaces
# 
# A subspace of a vector space is a subset that is itself a vector space under the same operations. To be a subspace, a subset must:
# 
# 1. Contain the zero vector
# 2. Be closed under vector addition: If u and v are in the subspace, u + v is also in the subspace
# 3. Be closed under scalar multiplication: If v is in the subspace and c is a scalar, c·v is also in the subspace
# 
# Let's visualize some subspaces in R³.

# %%
class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for visualization."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_subspaces_3d():
    """Plot different subspaces in R³."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define limits
    lim = 5
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    
    # Plot the axes
    ax.quiver(0, 0, 0, lim, 0, 0, color='k', arrow_length_ratio=0.1, alpha=0.5)
    ax.quiver(0, 0, 0, 0, lim, 0, color='k', arrow_length_ratio=0.1, alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, lim, color='k', arrow_length_ratio=0.1, alpha=0.5)
    
    ax.text(lim+0.2, 0, 0, "x", color='k')
    ax.text(0, lim+0.2, 0, "y", color='k')
    ax.text(0, 0, lim+0.2, "z", color='k')
    
    # Create a grid for various subspaces
    grid_size = 10
    x = np.linspace(-lim, lim, grid_size)
    y = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 1. A line through the origin (1D subspace) - span of a single vector
    v = np.array([3, 2, 1])
    line = np.outer(np.linspace(-1, 1, 100), v)
    ax.plot(line[:, 0], line[:, 1], line[:, 2], 'r-', linewidth=2, label='Line (1D subspace)')
    ax.text(v[0], v[1], v[2], "Span(v)", color='red')
    
    # 2. A plane through the origin (2D subspace) - span of two linearly independent vectors
    v1 = np.array([1, 0, 1])
    v2 = np.array([0, 1, 1])
    
    # Create a meshgrid for the plane
    Z = -X - Y  # This defines our plane: x + y + z = 0
    
    # Plot the plane
    ax.plot_surface(X, Y, Z, alpha=0.3, color='blue', label='Plane (2D subspace)')
    ax.text(lim/2, lim/2, -lim, "Span(v1, v2)", color='blue')
    
    # Add vectors that span the plane
    a1 = Arrow3D([0, v1[0]], [0, v1[1]], [0, v1[2]], mutation_scale=20, 
                 lw=2, arrowstyle='-|>', color='green')
    a2 = Arrow3D([0, v2[0]], [0, v2[1]], [0, v2[2]], mutation_scale=20, 
                 lw=2, arrowstyle='-|>', color='green')
    
    ax.add_artist(a1)
    ax.add_artist(a2)
    
    # 3. The origin (0D subspace)
    ax.scatter([0], [0], [0], color='purple', s=100, label='Origin (0D subspace)')
    
    # 4. The entire space (3D subspace)
    # This is R³ itself, the ambient space
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vector Subspaces in R³')
    
    # Create a custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Patch(facecolor='blue', alpha=0.3),
        Line2D([0], [0], marker='o', color='purple', markersize=10, linestyle='None')
    ]
    ax.legend(custom_lines, ['Line (1D)', 'Plane (2D)', 'Origin (0D)'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

plot_subspaces_3d()

# %% [markdown]
# The visualization above shows three types of subspaces in R³:
# 
# 1. **Point (0D)**: Just the origin (purple dot)
# 2. **Line (1D)**: A line passing through the origin (red line)
# 3. **Plane (2D)**: A plane passing through the origin (blue surface)
# 
# The entire R³ space itself would be a 3D subspace.
# 
# ## Span of Vectors
# 
# The span of a set of vectors is the set of all possible linear combinations of those vectors. In other words, the span is the subspace generated by those vectors.

# %%
def visualize_span_2d(vectors):
    """Visualize the span of vectors in 2D."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Convert PyTorch tensors to NumPy for easier plotting
    vectors_np = [v.numpy() if isinstance(v, torch.Tensor) else v for v in vectors]
    
    # Determine if vectors span a line or plane in R²
    if len(vectors) >= 2:
        # Check if vectors are linearly independent
        if len(vectors) == 2:
            v1, v2 = vectors_np
            det = v1[0] * v2[1] - v1[1] * v2[0]
            
            if abs(det) > 1e-10:  # Linearly independent
                # Vectors span the entire R² plane
                max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors_np]) * 1.5
                
                # Create a shaded area representing R²
                plt.fill_between([-max_val, max_val], -max_val, max_val, color='lightblue', alpha=0.3)
                plt.text(0, 0, "Span = R²", fontsize=14, ha='center', color='navy')
            else:  # Linearly dependent
                # Vectors span a line
                # Determine the direction of the line
                if all(v[0] == 0 for v in vectors_np):  # Vertical line
                    plt.axvline(x=0, color='lightblue', alpha=0.5, linewidth=5)
                    plt.text(0, 0, "Span = Line", fontsize=14, ha='center', color='navy')
                elif all(v[1] == 0 for v in vectors_np):  # Horizontal line
                    plt.axhline(y=0, color='lightblue', alpha=0.5, linewidth=5)
                    plt.text(0, 0, "Span = Line", fontsize=14, ha='center', color='navy')
                else:
                    # Find the non-zero vector to determine line direction
                    non_zero = next((v for v in vectors_np if not np.allclose(v, 0)), np.array([1.0, 1.0]))
                    
                    # Create extended line
                    max_val = max(abs(non_zero[0]), abs(non_zero[1])) * 5
                    t = np.linspace(-max_val, max_val, 100)
                    
                    # Normalize the direction
                    norm = np.sqrt(non_zero[0]**2 + non_zero[1]**2)
                    direction = non_zero / norm
                    
                    plt.plot(t * direction[0], t * direction[1], 'lightblue', alpha=0.5, linewidth=5)
                    plt.text(0, 0, "Span = Line", fontsize=14, ha='center', color='navy')
        else:
            # For more than 2 vectors, we don't check linear independence, just assume R²
            max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors_np]) * 1.5
            plt.fill_between([-max_val, max_val], -max_val, max_val, color='lightblue', alpha=0.3)
            plt.text(0, 0, "Span = R²", fontsize=14, ha='center', color='navy')
    else:  # Single vector case
        if len(vectors) == 1 and not np.allclose(vectors_np[0], 0):
            # Single non-zero vector spans a line
            v = vectors_np[0]
            max_val = max(abs(v[0]), abs(v[1])) * 5
            t = np.linspace(-max_val, max_val, 100)
            
            # Normalize the direction
            norm = np.sqrt(v[0]**2 + v[1]**2)
            direction = v / norm
            
            plt.plot(t * direction[0], t * direction[1], 'lightblue', alpha=0.5, linewidth=5)
            plt.text(0, 0, "Span = Line", fontsize=14, ha='center', color='navy')
        else:
            # Zero vector spans only the origin
            plt.scatter([0], [0], s=100, color='lightblue')
            plt.text(0, 0, "Span = {0}", fontsize=14, ha='center', color='navy')
    
    # Draw the vectors
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors_np]) * 1.2
    
    for i, v in enumerate(vectors_np):
        if not np.allclose(v, 0):  # Avoid drawing zero vector
            plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, 
                      head_length=max_val*0.08, fc=colors[i % len(colors)], 
                      ec=colors[i % len(colors)], label=f"v{i+1}")
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Span of Vectors in R²')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Case 1: Single vector (spans a line)
visualize_span_2d([torch.tensor([2.0, 1.0])])

# Case 2: Two linearly independent vectors (span R²)
visualize_span_2d([torch.tensor([2.0, 1.0]), torch.tensor([0.0, 3.0])])

# Case 3: Two linearly dependent vectors (span a line)
visualize_span_2d([torch.tensor([2.0, 1.0]), torch.tensor([4.0, 2.0])])

# Case 4: Zero vector (spans only the origin)
visualize_span_2d([torch.tensor([0.0, 0.0])])

# %% [markdown]
# The visualizations above demonstrate the concept of span:
# 
# 1. A single non-zero vector spans a line through the origin.
# 2. Two linearly independent vectors in R² span the entire R² space.
# 3. Two linearly dependent vectors span a line through the origin.
# 4. The zero vector spans only the origin itself.
# 
# ## Conclusion
# 
# In this notebook, we've introduced the fundamental concepts of vector spaces and subspaces:
# 
# - Vector spaces are collections of vectors that are closed under addition and scalar multiplication
# - Subspaces are subsets of vector spaces that are themselves vector spaces
# - The span of a set of vectors is the set of all possible linear combinations of those vectors
# 
# In the next notebook, we'll explore the concepts of basis, orthogonality, and projection in vector spaces.