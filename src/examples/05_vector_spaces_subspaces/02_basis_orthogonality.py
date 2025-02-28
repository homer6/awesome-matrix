# %% [markdown]
# # Vector Spaces: Basis, Orthogonality, and Projection
# 
# In this notebook, we continue our exploration of vector spaces by discussing basis vectors, orthogonality, and vector projections. These concepts are fundamental for understanding many applications in linear algebra.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("whitegrid")

# For better looking plots
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Basis Vectors
# 
# A basis for a vector space is a set of linearly independent vectors that span the entire space. In other words, every vector in the space can be written as a unique linear combination of the basis vectors.
# 
# The standard basis for R² consists of the vectors e₁ = (1,0) and e₂ = (0,1). Let's visualize this basis and see how other vectors can be represented in terms of it.

# %%
def plot_basis_vectors(basis_vectors, vectors_to_represent=None):
    """Plot basis vectors and optionally show how other vectors can be represented in this basis."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Ensure basis vectors are numpy arrays
    basis_vectors = [v.numpy() if isinstance(v, torch.Tensor) else v for v in basis_vectors]
    
    # Determine plot limits
    all_vectors = basis_vectors.copy()
    if vectors_to_represent:
        vectors_to_represent = [v.numpy() if isinstance(v, torch.Tensor) else v for v in vectors_to_represent]
        all_vectors.extend(vectors_to_represent)
    
    max_val = max([max(abs(v[0]), abs(v[1])) for v in all_vectors]) * 1.5
    
    # Plot grid using the basis vectors
    if len(basis_vectors) == 2:
        grid_lines = 5
        for i in range(-grid_lines, grid_lines + 1):
            # Draw horizontal grid lines (parallel to basis vector 1)
            plt.plot([-max_val, max_val], [i*basis_vectors[1][1], i*basis_vectors[1][1]], 
                     'gray', alpha=0.2)
            # Draw vertical grid lines (parallel to basis vector 2)
            plt.plot([i*basis_vectors[0][0], i*basis_vectors[0][0]], [-max_val, max_val], 
                     'gray', alpha=0.2)
    
    # Plot basis vectors
    colors = ['blue', 'red']
    for i, v in enumerate(basis_vectors):
        plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, head_length=max_val*0.08, 
                  fc=colors[i], ec=colors[i], label=f"Basis {i+1}")
    
    # Plot vectors to represent
    if vectors_to_represent:
        for i, v in enumerate(vectors_to_represent):
            # Plot the vector
            plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, head_length=max_val*0.08, 
                      fc='green', ec='green', label=f"Vector {i+1}")
            
            # If we have a 2D basis, show the representation
            if len(basis_vectors) == 2:
                # Solve the linear system to find coefficients
                A = np.column_stack(basis_vectors)
                coeffs = np.linalg.solve(A, v)
                
                # Plot components along basis vectors
                comp1 = coeffs[0] * np.array(basis_vectors[0])
                comp2 = coeffs[1] * np.array(basis_vectors[1])
                
                # Draw dashed lines for the components
                plt.plot([0, comp1[0]], [0, comp1[1]], 'b--', alpha=0.5)
                plt.plot([comp1[0], v[0]], [comp1[1], v[1]], 'r--', alpha=0.5)
                
                # Annotate with coefficients
                plt.text(comp1[0]/2, comp1[1]/2, f"{coeffs[0]:.2f}", color='blue')
                plt.text(comp1[0] + comp2[0]/2, comp1[1] + comp2[1]/2, f"{coeffs[1]:.2f}", color='red')
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(False)  # Turn off default grid since we're drawing our own
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Basis Vectors and Vector Representation')
    plt.legend()
    plt.show()

# Standard basis for R²
e1 = np.array([1, 0])
e2 = np.array([0, 1])
standard_basis = [e1, e2]

# A vector to represent
v = np.array([3, 2])

plot_basis_vectors(standard_basis, [v])

# %% [markdown]
# The plot above shows the standard basis vectors for R² (blue and red arrows). The green vector is represented as a linear combination of these basis vectors. Specifically, v = 3e₁ + 2e₂.
# 
# Let's try a different basis.

# %%
# A different basis for R²
b1 = np.array([1, 1])
b2 = np.array([-1, 1])
different_basis = [b1, b2]

plot_basis_vectors(different_basis, [v])

# %% [markdown]
# In this different basis, the same vector v is represented as a different linear combination of the basis vectors.
# 
# ## Dimension and Basis Size
# 
# The dimension of a vector space is the number of vectors in any basis for that space. For instance, R² has dimension 2, R³ has dimension 3, and so on.
# 
# Let's visualize the concept of a basis in R³.

# %%
def plot_basis_vectors_3d(basis_vectors, vectors_to_represent=None):
    """Plot basis vectors and optionally show how other vectors are represented in 3D."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ensure basis vectors are numpy arrays
    basis_vectors = [v.numpy() if isinstance(v, torch.Tensor) else v for v in basis_vectors]
    
    # Determine plot limits
    all_vectors = basis_vectors.copy()
    if vectors_to_represent:
        vectors_to_represent = [v.numpy() if isinstance(v, torch.Tensor) else v for v in vectors_to_represent]
        all_vectors.extend(vectors_to_represent)
    
    max_val = max([max(abs(v[0]), abs(v[1]), abs(v[2])) for v in all_vectors]) * 1.5
    lim = max_val
    
    # Plot basis vectors
    colors = ['blue', 'red', 'green']
    for i, v in enumerate(basis_vectors):
        ax.quiver(0, 0, 0, v[0], v[1], v[2], color=colors[i], arrow_length_ratio=0.1, 
                  label=f"Basis {i+1}")
    
    # Plot vectors to represent
    if vectors_to_represent:
        for i, v in enumerate(vectors_to_represent):
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color='purple', arrow_length_ratio=0.1, 
                      label=f"Vector {i+1}")
            
            # If we have a 3D basis, show the representation
            if len(basis_vectors) == 3:
                # Solve the linear system to find coefficients
                A = np.column_stack(basis_vectors)
                coeffs = np.linalg.solve(A, v)
                
                # Plot components along basis vectors as transparent cuboid
                origin = np.zeros(3)
                comp1 = coeffs[0] * np.array(basis_vectors[0])
                comp2 = coeffs[1] * np.array(basis_vectors[1])
                comp3 = coeffs[2] * np.array(basis_vectors[2])
                
                # We'll draw lines to show the decomposition
                # First comp1
                ax.plot([0, comp1[0]], [0, comp1[1]], [0, comp1[2]], 'b--', alpha=0.5)
                
                # Then add comp2
                p1 = comp1
                p2 = comp1 + comp2
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'r--', alpha=0.5)
                
                # Then add comp3
                p2 = comp1 + comp2
                p3 = comp1 + comp2 + comp3
                ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]], 'g--', alpha=0.5)
    
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Basis Vectors and Vector Representation')
    ax.legend()
    plt.show()

# Standard basis for R³
e1_3d = np.array([1, 0, 0])
e2_3d = np.array([0, 1, 0])
e3_3d = np.array([0, 0, 1])
standard_basis_3d = [e1_3d, e2_3d, e3_3d]

# A vector to represent in R³
v_3d = np.array([2, 1, 3])

plot_basis_vectors_3d(standard_basis_3d, [v_3d])

# %% [markdown]
# The 3D plot shows the standard basis for R³, consisting of the three unit vectors along the coordinate axes. The purple vector is represented as a linear combination of these basis vectors.
# 
# ## Orthogonality
# 
# Two vectors are orthogonal (perpendicular) if their dot product is zero. Orthogonal vectors are particularly useful in creating basis sets that are easy to work with.
# 
# Let's explore orthogonality and orthogonal bases.

# %%
def plot_orthogonal_vectors(vectors):
    """Plot vectors and check their orthogonality."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Ensure vectors are numpy arrays
    vectors = [v.numpy() if isinstance(v, torch.Tensor) else v for v in vectors]
    
    # Determine plot limits
    max_val = max([max(abs(v[0]), abs(v[1])) for v in vectors]) * 1.5
    
    # Plot vectors
    colors = ['blue', 'red', 'green', 'purple']
    for i, v in enumerate(vectors):
        plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, head_length=max_val*0.08, 
                  fc=colors[i % len(colors)], ec=colors[i % len(colors)], label=f"Vector {i+1}")
    
    # Check and display dot products
    n = len(vectors)
    dot_products = []
    for i in range(n):
        for j in range(i+1, n):
            dot_product = np.dot(vectors[i], vectors[j])
            normalized_dot = dot_product / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
            dot_products.append((i, j, dot_product, normalized_dot))
    
    # Add right angle markers for orthogonal vectors
    for i, j, dot, norm_dot in dot_products:
        if abs(dot) < 1e-10:  # Orthogonal
            # Find the point where to draw the right angle marker
            # Scale to be proportional to vector lengths
            scale = min(np.linalg.norm(vectors[i]), np.linalg.norm(vectors[j])) * 0.2
            
            # Draw L-shaped right angle marker
            plt.plot([vectors[i][0] * scale, 0, vectors[j][0] * scale], 
                     [vectors[i][1] * scale, 0, vectors[j][1] * scale], 
                     'k-', linewidth=1)
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Orthogonal Vectors')
    plt.legend()
    
    # Add inset with dot products
    plt.axis('on')
    textstr = '\n'.join([f"v{i+1}·v{j+1} = {dot:.2f}" + 
                         f" (normalized: {norm_dot:.2f})" for i, j, dot, norm_dot in dot_products])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props)
    
    plt.show()

# Standard basis (orthogonal vectors)
v1 = np.array([1, 0])
v2 = np.array([0, 1])

# Non-orthogonal vectors
v3 = np.array([1, 1])
v4 = np.array([1, -0.5])

plot_orthogonal_vectors([v1, v2])
plot_orthogonal_vectors([v3, v4])

# %% [markdown]
# The first plot shows two orthogonal vectors (the standard basis), while the second plot shows two non-orthogonal vectors. We can verify orthogonality by checking that the dot product is zero.
# 
# An orthogonal basis is a basis where all vectors are orthogonal to each other. An orthonormal basis is an orthogonal basis where all basis vectors have unit length (norm of 1).

# %%
def plot_orthonormal_basis():
    """Plot an orthonormal basis for R²."""
    # Standard orthonormal basis
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    # Create a rotated orthonormal basis
    theta = np.pi/4  # 45 degrees rotation
    r1 = np.array([np.cos(theta), np.sin(theta)])
    r2 = np.array([-np.sin(theta), np.cos(theta)])
    
    plt.figure(figsize=(12, 6))
    
    # Plot standard basis
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.arrow(0, 0, e1[0], e1[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='e₁')
    plt.arrow(0, 0, e2[0], e2[1], head_width=0.1, head_length=0.1, fc='red', ec='red', label='e₂')
    
    # Draw circle to show unit vectors
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    plt.plot(circle_x, circle_y, 'k--', alpha=0.3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Standard Orthonormal Basis')
    plt.legend()
    
    # Plot rotated basis
    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.arrow(0, 0, r1[0], r1[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='r₁')
    plt.arrow(0, 0, r2[0], r2[1], head_width=0.1, head_length=0.1, fc='red', ec='red', label='r₂')
    
    # Draw circle to show unit vectors
    plt.plot(circle_x, circle_y, 'k--', alpha=0.3)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Rotated Orthonormal Basis')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Verify orthogonality
    print(f"Dot product of standard basis vectors: {np.dot(e1, e2):.10f}")
    print(f"Dot product of rotated basis vectors: {np.dot(r1, r2):.10f}")
    
    # Verify unit length
    print(f"Norm of e₁: {np.linalg.norm(e1):.10f}")
    print(f"Norm of e₂: {np.linalg.norm(e2):.10f}")
    print(f"Norm of r₁: {np.linalg.norm(r1):.10f}")
    print(f"Norm of r₂: {np.linalg.norm(r2):.10f}")

plot_orthonormal_basis()

# %% [markdown]
# The plots above show two different orthonormal bases for R²: the standard basis and a rotated basis (rotated by 45 degrees). Both sets of basis vectors are orthogonal to each other (dot product = 0) and have unit length (norm = 1).
# 
# ## Vector Projection
# 
# Projection is a fundamental operation in vector spaces. It allows us to find the component of one vector that points in the direction of another.
# 
# If we have two vectors a and b, the projection of a onto b is:
# 
# $$\text{proj}_b(a) = \frac{a \cdot b}{b \cdot b} \cdot b = \frac{a \cdot b}{\|b\|^2} \cdot b$$
# 
# Let's visualize vector projections.

# %%
def plot_vector_projection(a, b):
    """Plot vector projection of a onto b."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Ensure vectors are numpy arrays
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    b = b.numpy() if isinstance(b, torch.Tensor) else b
    
    # Calculate projection
    scalar_proj = np.dot(a, b) / np.dot(b, b)
    vector_proj = scalar_proj * b
    
    # Calculate orthogonal component (a - proj)
    orth_component = a - vector_proj
    
    # Determine plot limits
    max_val = max(max(abs(a[0]), abs(a[1])), max(abs(b[0]), abs(b[1]))) * 1.5
    
    # Plot vectors
    plt.arrow(0, 0, a[0], a[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='blue', ec='blue', label='a')
    plt.arrow(0, 0, b[0], b[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='red', ec='red', label='b')
    
    # Plot projection vector
    plt.arrow(0, 0, vector_proj[0], vector_proj[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='green', ec='green', label='proj_b(a)')
    
    # Plot orthogonal component
    plt.arrow(0, 0, orth_component[0], orth_component[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='purple', ec='purple', label='a - proj_b(a)')
    
    # Draw dashed line from a to projection
    plt.plot([a[0], vector_proj[0]], [a[1], vector_proj[1]], 'k--', alpha=0.5)
    
    # Draw a right angle marker between projection and orthogonal component
    # to show they are perpendicular
    scale = min(np.linalg.norm(vector_proj), np.linalg.norm(orth_component)) * 0.2
    if np.linalg.norm(vector_proj) > 1e-10 and np.linalg.norm(orth_component) > 1e-10:
        # Normalize vectors for the right angle marker
        v_norm = vector_proj / np.linalg.norm(vector_proj)
        o_norm = orth_component / np.linalg.norm(orth_component)
        
        plt.plot([0, v_norm[0] * scale, v_norm[0] * scale + o_norm[0] * scale], 
                 [0, v_norm[1] * scale, v_norm[1] * scale + o_norm[1] * scale], 
                 'k-', linewidth=1)
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Vector Projection')
    plt.legend()
    
    # Add projection details
    textstr = f"Scalar projection: {scalar_proj:.2f}\n"
    textstr += f"Projection of a onto b: [{vector_proj[0]:.2f}, {vector_proj[1]:.2f}]\n"
    textstr += f"Orthogonal component: [{orth_component[0]:.2f}, {orth_component[1]:.2f}]"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='bottom', bbox=props)
    
    plt.show()

# Example vectors
a = np.array([4, 3])
b = np.array([2, 0])

plot_vector_projection(a, b)

# %% [markdown]
# In the plot above:
# 
# - The blue vector is a
# - The red vector is b
# - The green vector is the projection of a onto b
# - The purple vector is the orthogonal component (a - projection)
# 
# Note that the green and purple vectors form a right angle, demonstrating that the projection and the orthogonal component are perpendicular to each other.
# 
# Let's try with different vectors.

# %%
# Different example vectors
a2 = np.array([3, 2])
b2 = np.array([1, 2])

plot_vector_projection(a2, b2)

# %% [markdown]
# ## Applications: Change of Basis
# 
# One important application of these concepts is changing the basis of a vector. This allows us to represent the same vector in different coordinate systems.
# 
# Let's visualize how a vector's coordinates change when we switch from one basis to another.

# %%
def plot_change_of_basis(v, basis1, basis2):
    """Plot a vector in two different bases."""
    plt.figure(figsize=(12, 6))
    
    # Ensure vectors are numpy arrays
    v = v.numpy() if isinstance(v, torch.Tensor) else v
    basis1 = [b.numpy() if isinstance(b, torch.Tensor) else b for b in basis1]
    basis2 = [b.numpy() if isinstance(b, torch.Tensor) else b for b in basis2]
    
    # Calculate the coordinates in each basis
    A1 = np.column_stack(basis1)
    coords1 = np.linalg.solve(A1, v)
    
    A2 = np.column_stack(basis2)
    coords2 = np.linalg.solve(A2, v)
    
    # Plot vector in first basis
    plt.subplot(1, 2, 1)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Determine plot limits
    max_val = max(max(abs(v[0]), abs(v[1])), 
                 max(max(abs(basis1[0][0]), abs(basis1[0][1])), max(abs(basis1[1][0]), abs(basis1[1][1])))) * 1.5
    
    # Plot basis vectors
    plt.arrow(0, 0, basis1[0][0], basis1[0][1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='blue', ec='blue', label='Basis 1, vector 1')
    plt.arrow(0, 0, basis1[1][0], basis1[1][1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='red', ec='red', label='Basis 1, vector 2')
    
    # Plot the vector
    plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='green', ec='green', label='Vector v')
    
    # Draw components in first basis
    comp1 = coords1[0] * np.array(basis1[0])
    comp2 = coords1[1] * np.array(basis1[1])
    
    plt.plot([0, comp1[0]], [0, comp1[1]], 'b--', alpha=0.5)
    plt.plot([comp1[0], v[0]], [comp1[1], v[1]], 'r--', alpha=0.5)
    
    # Annotate with coefficients
    plt.text(comp1[0]/2, comp1[1]/2, f"{coords1[0]:.2f}", color='blue')
    plt.text(comp1[0] + comp2[0]/2, comp1[1] + comp2[1]/2, f"{coords1[1]:.2f}", color='red')
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Vector in Basis 1: [{coords1[0]:.2f}, {coords1[1]:.2f}]')
    plt.legend()
    
    # Plot vector in second basis
    plt.subplot(1, 2, 2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Determine plot limits
    max_val = max(max(abs(v[0]), abs(v[1])), 
                 max(max(abs(basis2[0][0]), abs(basis2[0][1])), max(abs(basis2[1][0]), abs(basis2[1][1])))) * 1.5
    
    # Plot basis vectors
    plt.arrow(0, 0, basis2[0][0], basis2[0][1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='blue', ec='blue', label='Basis 2, vector 1')
    plt.arrow(0, 0, basis2[1][0], basis2[1][1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='red', ec='red', label='Basis 2, vector 2')
    
    # Plot the vector
    plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='green', ec='green', label='Vector v')
    
    # Draw components in second basis
    comp1 = coords2[0] * np.array(basis2[0])
    comp2 = coords2[1] * np.array(basis2[1])
    
    plt.plot([0, comp1[0]], [0, comp1[1]], 'b--', alpha=0.5)
    plt.plot([comp1[0], v[0]], [comp1[1], v[1]], 'r--', alpha=0.5)
    
    # Annotate with coefficients
    plt.text(comp1[0]/2, comp1[1]/2, f"{coords2[0]:.2f}", color='blue')
    plt.text(comp1[0] + comp2[0]/2, comp1[1] + comp2[1]/2, f"{coords2[1]:.2f}", color='red')
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Vector in Basis 2: [{coords2[0]:.2f}, {coords2[1]:.2f}]')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Vector to represent
v = np.array([3, 2])

# Standard basis
standard_basis = [np.array([1, 0]), np.array([0, 1])]

# Rotated basis (45 degrees)
theta = np.pi/4
rotated_basis = [np.array([np.cos(theta), np.sin(theta)]), 
                np.array([-np.sin(theta), np.cos(theta)])]

plot_change_of_basis(v, standard_basis, rotated_basis)

# %% [markdown]
# The plots show the same vector v represented in two different bases:
# 
# 1. The standard basis, where v = [3.00, 2.00]
# 2. A rotated basis (45 degrees), where the coordinates of v are different
# 
# This demonstrates that the coordinates of a vector depend on the choice of basis.
# 
# ## Change of Basis Matrix
# 
# We can compute a change of basis matrix that transforms coordinates from one basis to another. If we have basis B and basis C, the change of basis matrix from B to C is:
# 
# $$P_{B \to C} = [c_1, c_2, \ldots, c_n]^{-1} \cdot [b_1, b_2, \ldots, b_n]$$
# 
# where $b_i$ and $c_i$ are the basis vectors, expressed in the standard basis.

# %%
def compute_change_of_basis_matrix(basis_from, basis_to):
    """Compute the change of basis matrix from basis_from to basis_to."""
    # Each basis is a list of column vectors
    # Convert to numpy arrays
    basis_from = [b.numpy() if isinstance(b, torch.Tensor) else b for b in basis_from]
    basis_to = [b.numpy() if isinstance(b, torch.Tensor) else b for b in basis_to]
    
    # Create matrices where columns are basis vectors
    B = np.column_stack(basis_from)
    C = np.column_stack(basis_to)
    
    # Change of basis matrix from B to C
    P = np.linalg.inv(C) @ B
    
    return P

# Standard basis
standard_basis = [np.array([1, 0]), np.array([0, 1])]

# Rotated basis (45 degrees)
theta = np.pi/4
rotated_basis = [np.array([np.cos(theta), np.sin(theta)]), 
                np.array([-np.sin(theta), np.cos(theta)])]

# Compute change of basis matrix
P_standard_to_rotated = compute_change_of_basis_matrix(standard_basis, rotated_basis)
P_rotated_to_standard = compute_change_of_basis_matrix(rotated_basis, standard_basis)

print("Change of basis matrix from standard to rotated:")
print(P_standard_to_rotated)
print("\nChange of basis matrix from rotated to standard:")
print(P_rotated_to_standard)

# Test with our vector
v_standard = np.array([3, 2])
v_rotated = P_standard_to_rotated @ v_standard
v_back_to_standard = P_rotated_to_standard @ v_rotated

print("\nVector in standard basis:", v_standard)
print("Vector in rotated basis:", v_rotated)
print("Vector converted back to standard basis:", v_back_to_standard)

# %% [markdown]
# The change of basis matrices allow us to convert coordinates from one basis to another. We can verify that the conversion works correctly by converting a vector from standard to rotated basis, and then back to standard, which should give us the original vector.
# 
# ## Orthogonal Projections in Subspaces
# 
# Orthogonal projection is a key concept in the theory of vector spaces. It allows us to find the closest vector in a subspace to a given vector.
# 
# Let's visualize the projection of a vector onto a subspace (in this case, a line).

# %%
def plot_projection_onto_subspace(v, subspace_basis):
    """Plot the projection of a vector onto a subspace."""
    plt.figure(figsize=(10, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Ensure vectors are numpy arrays
    v = v.numpy() if isinstance(v, torch.Tensor) else v
    subspace_basis = [b.numpy() if isinstance(b, torch.Tensor) else b for b in subspace_basis]
    
    # Create the projection matrix
    # P = B(B^T B)^(-1)B^T where B is the matrix whose columns are the basis vectors
    B = np.column_stack(subspace_basis)
    P = B @ np.linalg.inv(B.T @ B) @ B.T
    
    # Calculate the projection
    v_proj = P @ v
    
    # Calculate orthogonal component (v - v_proj)
    v_orth = v - v_proj
    
    # Determine plot limits
    all_vectors = subspace_basis + [v, v_proj, v_orth]
    max_val = max([max(abs(vec[0]), abs(vec[1])) for vec in all_vectors]) * 1.5
    
    # Plot subspace (line or plane)
    if len(subspace_basis) == 1:
        # Subspace is a line
        t = np.linspace(-max_val, max_val, 100)
        basis_vec = subspace_basis[0]
        unit_vec = basis_vec / np.linalg.norm(basis_vec)
        plt.plot(t * unit_vec[0], t * unit_vec[1], 'gray', alpha=0.3, label='Subspace')
    elif len(subspace_basis) == 2:
        # Subspace is a plane (the entire R²)
        # We'll just indicate this in the legend
        pass
    
    # Plot basis vectors
    for i, basis_vec in enumerate(subspace_basis):
        plt.arrow(0, 0, basis_vec[0], basis_vec[1], head_width=max_val*0.05, head_length=max_val*0.08, 
                  fc='blue', ec='blue', label=f'Basis vector {i+1}')
    
    # Plot the original vector
    plt.arrow(0, 0, v[0], v[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='red', ec='red', label='Vector v')
    
    # Plot the projection
    plt.arrow(0, 0, v_proj[0], v_proj[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='green', ec='green', label='Projection')
    
    # Plot the orthogonal component
    plt.arrow(0, 0, v_orth[0], v_orth[1], head_width=max_val*0.05, head_length=max_val*0.08, 
              fc='purple', ec='purple', label='Orthogonal component')
    
    # Draw dashed line from v to projection
    plt.plot([v[0], v_proj[0]], [v[1], v_proj[1]], 'k--', alpha=0.5)
    
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Projection onto a Subspace')
    plt.legend()
    plt.show()

# Vector to project
v = np.array([3, 2])

# Subspace basis (1D subspace = line)
subspace_basis = [np.array([1, 1])]

plot_projection_onto_subspace(v, subspace_basis)

# %% [markdown]
# The plot shows:
# 
# - The original vector v (red)
# - The subspace, which is a line in this case (gray)
# - The projection of v onto the subspace (green)
# - The orthogonal component (purple)
# 
# The projection is the closest vector to v that lies in the subspace. The orthogonal component is perpendicular to the subspace.
# 
# Let's try projection onto a different subspace.

# %%
# Different subspace (another line)
subspace_basis2 = [np.array([0, 1])]

plot_projection_onto_subspace(v, subspace_basis2)

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored key concepts in vector spaces:
# 
# - Basis vectors and how they define a coordinate system
# - Orthogonality and orthonormal bases
# - Vector projections and orthogonal components
# - Change of basis and coordinate transformations
# - Projection onto subspaces
# 
# These concepts are fundamental in many areas of mathematics, physics, computer graphics, machine learning, and data analysis. Understanding these geometric interpretations helps build intuition for more complex linear algebra operations.
# 
# In the next notebook, we'll explore the concept of change of basis in more detail and explore applications in transformations and coordinate systems.