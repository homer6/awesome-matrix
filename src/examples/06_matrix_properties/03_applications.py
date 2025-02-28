# %% [markdown]
# # Matrix Properties: Applications
# 
# In this notebook, we explore real-world applications of matrix properties. Understanding how matrix properties apply to practical problems helps develop intuition for why they matter in fields like machine learning, computer graphics, data analysis, and scientific computing.
# 
# We'll focus on the following applications:
# 
# 1. **Image Processing and Compression**
# 2. **Principal Component Analysis (PCA)**
# 3. **Linear Systems and Condition Number**
# 4. **Graph Analysis with Adjacency Matrices**
# 5. **Markov Chains and Transition Matrices**
# 
# Each application demonstrates how matrix properties inform algorithm design and practical solutions.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import requests
from io import BytesIO
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits, fetch_olivetti_faces, make_blobs
import networkx as nx
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import StandardScaler

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
# ## 1. Image Processing and Compression
# 
# Images can be represented as matrices where each element corresponds to a pixel value. Many image processing tasks involve matrix operations, such as:
# 
# - **Filtering and convolution**: Using small matrices (kernels) to transform images
# - **Compression**: Reducing the storage requirements of images using matrix factorization
# - **Feature extraction**: Identifying important patterns in images
# 
# Let's explore how matrix rank and Singular Value Decomposition (SVD) can be used for image compression.

# %%
def load_grayscale_image(url=None):
    """Load a grayscale image from a URL or use a default image."""
    if url is None:
        # Default image URL (Claude Shannon)
        url = "https://upload.wikimedia.org/wikipedia/commons/9/9b/Claude_Shannon_MF.jpg"
    
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
        # Resize for faster processing
        img = img.resize((256, 256))
        return np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        # Create a simple synthetic image
        x = np.linspace(0, 1, 256)
        y = np.linspace(0, 1, 256)
        X, Y = np.meshgrid(x, y)
        synthetic_image = (255 * (np.sin(10*X) * np.sin(10*Y))).astype(np.uint8)
        return synthetic_image

# Load a grayscale image
image = load_grayscale_image()

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.show()

# %% [markdown]
# ### Image Compression with SVD
# 
# Singular Value Decomposition (SVD) is a powerful matrix factorization technique that can be used for image compression. The SVD of a matrix $A$ is given by:
# 
# $A = U \Sigma V^T$
# 
# where:
# - $U$ is an orthogonal matrix of left singular vectors
# - $\Sigma$ is a diagonal matrix of singular values
# - $V^T$ is the transpose of an orthogonal matrix of right singular vectors
# 
# By keeping only the largest singular values and their corresponding singular vectors, we can create a low-rank approximation of the original image.

# %%
def svd_image_compression(image, ranks_to_test):
    """Compress an image using SVD with different rank approximations."""
    # Perform SVD
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Calculate the storage requirements
    original_size = image.size
    
    # Create the compressed images
    compressed_images = []
    compression_ratios = []
    
    for r in ranks_to_test:
        # Low-rank approximation
        compressed = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        compressed_images.append(compressed)
        
        # Calculate compression ratio
        compressed_size = r * (U.shape[0] + Vt.shape[1] + 1)  # +1 for singular value
        compression_ratio = original_size / compressed_size
        compression_ratios.append(compression_ratio)
    
    return compressed_images, compression_ratios, S

# Define ranks for compression
ranks = [1, 5, 10, 20, 50, 100]

# Compress the image
compressed_images, compression_ratios, singular_values = svd_image_compression(image, ranks)

# Display original and compressed images
plt.figure(figsize=(15, 12))

# Original
plt.subplot(2, 4, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.axis('off')

# Compressed images
for i, (r, img, ratio) in enumerate(zip(ranks, compressed_images, compression_ratios)):
    plt.subplot(2, 4, i + 2)
    plt.imshow(img, cmap='gray')
    plt.title(f"Rank {r}\nCompression Ratio: {ratio:.1f}:1")
    plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analysis of Singular Values
# 
# Let's analyze the singular values to understand how much information is captured by each rank approximation.

# %%
def analyze_singular_values(S):
    """Analyze and visualize singular values."""
    # Total energy (sum of all singular values squared)
    total_energy = np.sum(S**2)
    
    # Calculate cumulative energy percentage
    cumulative_energy = np.cumsum(S**2) / total_energy * 100
    
    # Plot singular values
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(S, 'o-', color='blue', markersize=4)
    plt.yscale('log')
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_energy, 'o-', color='green', markersize=4)
    plt.xlabel('Number of Singular Values')
    plt.ylabel('Cumulative Energy (%)')
    plt.title('Energy Captured vs. Rank')
    plt.grid(True, alpha=0.3)
    
    # Add threshold lines for reference
    thresholds = [90, 95, 99]
    for t in thresholds:
        # Find the rank needed to capture t% of the energy
        rank = np.where(cumulative_energy >= t)[0][0] + 1
        plt.axhline(y=t, color='red', linestyle='--', alpha=0.3)
        plt.axvline(x=rank, color='red', linestyle='--', alpha=0.3)
        plt.text(rank + 5, t - 2, f"{rank} singular values = {t}% energy", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("Singular Value Analysis:")
    for t in thresholds:
        rank = np.where(cumulative_energy >= t)[0][0] + 1
        print(f"{rank} singular values capture {t}% of the image information")
    
    return cumulative_energy

# Analyze singular values
cumulative_energy = analyze_singular_values(singular_values)

# %% [markdown]
# ### Visual Interpretation of SVD Components
# 
# Let's visualize how the image is constructed from the singular vectors. Each rank-1 component is the outer product of a left and right singular vector, weighted by the corresponding singular value.

# %%
def visualize_svd_components(image, n_components=5):
    """Visualize the top SVD components of an image."""
    # Perform SVD
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Visualize original image and top components
    plt.figure(figsize=(15, 8))
    
    # Original image
    plt.subplot(2, n_components + 1, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Cumulative reconstruction
    plt.subplot(2, n_components + 1, n_components + 2)
    cumulative = np.zeros_like(image, dtype=float)
    plt.imshow(cumulative, cmap='gray')
    plt.title("Rank 0")
    plt.axis('off')
    
    # Visualize individual components and cumulative reconstruction
    for i in range(n_components):
        # Rank-1 component from outer product of singular vectors
        component = np.outer(U[:, i], Vt[i, :]) * S[i]
        
        # Scale for visualization
        scaled_component = (component - component.min()) / (component.max() - component.min())
        
        # Display component
        plt.subplot(2, n_components + 1, i + 2)
        plt.imshow(scaled_component, cmap='gray')
        plt.title(f"Component {i+1}\nλ = {S[i]:.1f}")
        plt.axis('off')
        
        # Update cumulative reconstruction
        cumulative += component
        scaled_cumulative = np.clip(cumulative, 0, 255)
        
        # Display cumulative
        plt.subplot(2, n_components + 1, n_components + i + 3)
        plt.imshow(scaled_cumulative, cmap='gray')
        plt.title(f"Rank {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize SVD components
visualize_svd_components(image, n_components=5)

# %% [markdown]
# ## 2. Principal Component Analysis (PCA)
# 
# Principal Component Analysis (PCA) is a dimensionality reduction technique that uses matrix properties to identify the most informative directions in a dataset. PCA relies on eigendecomposition of the covariance matrix to find these principal components.
# 
# Let's apply PCA to visualize high-dimensional data and understand how it relates to matrix properties.

# %%
def load_digit_data():
    """Load the digits dataset for PCA demonstration."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    return X, y, digits.images[0].shape

# Load digits data
X, y, image_shape = load_digit_data()

# Standardize the data
X_std = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_std)

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_ * 100
cumulative_variance = np.cumsum(explained_variance)

# Show sample images
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[y == i][0].reshape(image_shape), cmap='gray')
    plt.title(f"Digit: {i}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize explained variance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, 11), explained_variance[:10], alpha=0.7, color='skyblue', edgecolor='blue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance (%)')
plt.title('Variance Explained by Each Principal Component')
plt.xticks(range(1, 11))
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', color='green', markersize=4)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance')
plt.grid(True, alpha=0.3)

# Add threshold lines
thresholds = [70, 80, 90, 95]
for t in thresholds:
    # Find the number of components needed to explain t% of the variance
    n_components = np.where(cumulative_variance >= t)[0][0] + 1
    plt.axhline(y=t, color='red', linestyle='--', alpha=0.3)
    plt.axvline(x=n_components, color='red', linestyle='--', alpha=0.3)
    plt.text(n_components + 1, t - 3, f"{n_components} PCs", fontsize=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualizing Principal Components
# 
# Let's visualize the first few principal components to understand what features they capture.

# %%
def visualize_principal_components(pca_model, image_shape, n_components=8):
    """Visualize principal components as images."""
    # Get the components
    components = pca_model.components_
    
    plt.figure(figsize=(12, 4))
    for i in range(n_components):
        plt.subplot(2, n_components // 2, i + 1)
        # Reshape component to image shape
        component = components[i].reshape(image_shape)
        
        # Scale for visualization
        vmax = max(abs(component.max()), abs(component.min()))
        plt.imshow(component, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.title(f"PC {i+1}\n({explained_variance[i]:.1f}%)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize principal components
visualize_principal_components(pca, image_shape)

# %% [markdown]
# ### Visualizing Data in Principal Component Space
# 
# Let's visualize the digits data in the space defined by the first two principal components.

# %%
def plot_pca_scatter(X_pca, y, n_classes=10):
    """Plot a scatter plot of data in PCA space."""
    plt.figure(figsize=(10, 8))
    
    # Create a colormap
    cmap = plt.cm.get_cmap('tab10', n_classes)
    
    # Plot each class with a different color
    for i in range(n_classes):
        plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                    alpha=0.7, color=cmap(i), label=f"Digit {i}")
    
    plt.xlabel(f"Principal Component 1 ({explained_variance[0]:.1f}%)")
    plt.ylabel(f"Principal Component 2 ({explained_variance[1]:.1f}%)")
    plt.title("Digits Dataset in PCA Space")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot PCA scatter plot
plot_pca_scatter(X_pca, y)

# %% [markdown]
# ### Reconstructing Data from Principal Components
# 
# We can also reconstruct the original data using a subset of principal components, which is useful for compression and denoising.

# %%
def reconstruct_from_pca(X, pca_model, n_components_list):
    """Reconstruct data using different numbers of principal components."""
    # Original data shape
    X_std = StandardScaler().fit_transform(X)
    
    # Transform to PCA space
    X_pca = pca_model.transform(X_std)
    
    # Sample a few digits for demonstration
    digit_indices = [np.where(y == i)[0][0] for i in range(10)]
    sample_digits = X_std[digit_indices]
    sample_pca = X_pca[digit_indices]
    
    # Visualize reconstructions
    reconstructions = []
    for n_components in n_components_list:
        # Zero out components we're not using
        reduced_sample = sample_pca.copy()
        reduced_sample[:, n_components:] = 0
        
        # Inverse transform to original space
        reconstruction = pca_model.inverse_transform(reduced_sample)
        reconstructions.append(reconstruction)
    
    # Plot
    plt.figure(figsize=(15, 12))
    
    # For each digit
    for i, idx in enumerate(range(10)):
        # Original
        plt.subplot(len(n_components_list) + 1, 10, i + 1)
        plt.imshow(sample_digits[i].reshape(image_shape), cmap='gray')
        if i == 0:
            plt.ylabel("Original", rotation=90, size=12)
        plt.axis('off')
        plt.title(f"Digit {idx}")
        
        # Reconstructions
        for j, (n_components, reconstruction) in enumerate(zip(n_components_list, reconstructions)):
            plt.subplot(len(n_components_list) + 1, 10, (j + 1) * 10 + i + 1)
            plt.imshow(reconstruction[i].reshape(image_shape), cmap='gray')
            if i == 0:
                plt.ylabel(f"{n_components} PCs", rotation=90, size=12)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Reconstruct digits with different numbers of principal components
n_components_list = [5, 10, 20, 40]
reconstruct_from_pca(X, pca, n_components_list)

# %% [markdown]
# ## 3. Linear Systems and Condition Number
# 
# The condition number of a matrix affects the stability of linear systems. A high condition number indicates that small changes in the input can cause large changes in the output, which can lead to numerical instability.
# 
# Let's explore how the condition number affects the solution of linear systems.

# %%
def create_matrices_with_varying_condition(sizes=[10, 50, 100, 200], random_state=42):
    """Create matrices with different condition numbers."""
    np.random.seed(random_state)
    
    matrices = []
    for size in sizes:
        # Create a random matrix
        A = np.random.rand(size, size)
        
        # Compute the SVD
        U, S, Vt = np.linalg.svd(A)
        
        # Create matrices with different condition numbers by manipulating singular values
        
        # Well-conditioned matrix (condition number ≈ 1)
        S_well = np.ones_like(S)
        A_well = U @ np.diag(S_well) @ Vt
        cond_well = np.linalg.cond(A_well)
        matrices.append(("Well-conditioned", A_well, cond_well, size))
        
        # Moderately ill-conditioned matrix
        S_moderate = np.logspace(0, 3, size)  # Linear decay
        A_moderate = U @ np.diag(S_moderate) @ Vt
        cond_moderate = np.linalg.cond(A_moderate)
        matrices.append(("Moderately ill-conditioned", A_moderate, cond_moderate, size))
        
        # Severely ill-conditioned matrix
        S_severe = np.logspace(0, 6, size)  # Exponential decay
        A_severe = U @ np.diag(S_severe) @ Vt
        cond_severe = np.linalg.cond(A_severe)
        matrices.append(("Severely ill-conditioned", A_severe, cond_severe, size))
    
    return matrices

# Create matrices with different condition numbers
matrices = create_matrices_with_varying_condition()

# Display condition numbers
print("Matrix Condition Numbers:")
for name, A, cond, size in matrices:
    print(f"{name} ({size}x{size}): {cond:.2e}")

# %% [markdown]
# ### Solving Linear Systems with Different Condition Numbers
# 
# Now let's see how the condition number affects the solution of linear systems when there's noise in the input.

# %%
def simulate_linear_system_with_noise(matrices, noise_levels=[0, 1e-10, 1e-8, 1e-6, 1e-4]):
    """Simulate solving linear systems with different noise levels."""
    results = []
    
    for name, A, cond, size in matrices:
        # Create a known solution
        x_true = np.ones(size)
        
        # Compute the right-hand side
        b = A @ x_true
        
        # Solve the system with different noise levels
        errors = []
        for noise_level in noise_levels:
            # Add noise to b
            noise = np.random.normal(0, noise_level, size)
            b_noisy = b + noise
            
            # Solve the system
            try:
                x_noisy = np.linalg.solve(A, b_noisy)
                
                # Compute relative error
                rel_error = np.linalg.norm(x_noisy - x_true) / np.linalg.norm(x_true)
                errors.append(rel_error)
            except np.linalg.LinAlgError:
                errors.append(np.nan)
        
        results.append((name, cond, size, errors))
    
    return results

# Simulate linear systems with noise
results = simulate_linear_system_with_noise(matrices[:9])  # Use the first 9 matrices

# Plot the results
plt.figure(figsize=(12, 8))

# Group by condition number type
for i, cond_type in enumerate(["Well-conditioned", "Moderately ill-conditioned", "Severely ill-conditioned"]):
    plt.subplot(1, 3, i+1)
    
    # Filter results for this condition type
    type_results = [r for r in results if r[0] == cond_type]
    
    # Plot for each matrix size
    noise_levels = [0, 1e-10, 1e-8, 1e-6, 1e-4]
    for name, cond, size, errors in type_results:
        plt.semilogy(range(len(noise_levels)), errors, 'o-', label=f"Size {size}, κ={cond:.2e}")
    
    plt.xlabel("Noise Level")
    plt.ylabel("Relative Error")
    plt.title(f"{cond_type} Matrices")
    plt.xticks(range(len(noise_levels)), [f"{n:.0e}" if n > 0 else "0" for n in noise_levels])
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Visualizing the Effect of Condition Number on a 2D System
# 
# For a 2D system, we can visualize how the condition number affects the solution geometrically.

# %%
def visualize_condition_number_2d():
    """Visualize the effect of condition number on a 2D linear system."""
    # Create matrices with different condition numbers
    np.random.seed(42)
    
    # Well-conditioned matrix
    A_well = np.array([[1.0, 0.2], 
                        [0.2, 1.0]])
    
    # Moderately ill-conditioned matrix
    A_moderate = np.array([[1.0, 0.99], 
                           [0.99, 1.0]])
    
    # Severely ill-conditioned matrix
    A_severe = np.array([[1.0, 0.999], 
                         [0.999, 1.0]])
    
    matrices = [
        ("Well-conditioned", A_well, np.linalg.cond(A_well)),
        ("Moderately ill-conditioned", A_moderate, np.linalg.cond(A_moderate)),
        ("Severely ill-conditioned", A_severe, np.linalg.cond(A_severe))
    ]
    
    # Define true solution
    x_true = np.array([1.0, 1.0])
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    for i, (name, A, cond) in enumerate(matrices):
        # Compute right-hand side
        b = A @ x_true
        
        # Create grid for visualization
        n = 100
        x1 = np.linspace(-1, 3, n)
        x2 = np.linspace(-1, 3, n)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Compute residual for each point on the grid
        residuals = np.zeros((n, n))
        for i1 in range(n):
            for i2 in range(n):
                x = np.array([X1[i1, i2], X2[i1, i2]])
                residuals[i1, i2] = np.linalg.norm(A @ x - b)
        
        # Plot
        plt.subplot(2, 3, i+1)
        
        # Contour plot of residuals
        contour = plt.contourf(X1, X2, residuals, 50, cmap='viridis', alpha=0.7)
        plt.colorbar(contour, label="Residual ||Ax - b||")
        
        # Plot true solution
        plt.scatter(x_true[0], x_true[1], color='red', marker='x', s=100, label="True Solution")
        
        # Add noise to b and solve
        noise_levels = [1e-6, 1e-4, 1e-2]
        for j, noise_level in enumerate(noise_levels):
            np.random.seed(j)  # Make it reproducible
            noise = np.random.normal(0, noise_level, 2)
            b_noisy = b + noise
            
            try:
                x_noisy = np.linalg.solve(A, b_noisy)
                plt.scatter(x_noisy[0], x_noisy[1], color=f'C{j+1}', marker='o', 
                            label=f"Noise {noise_level:.0e}")
                
                # Draw line connecting true and noisy solutions
                plt.plot([x_true[0], x_noisy[0]], [x_true[1], x_noisy[1]], 
                         color=f'C{j+1}', linestyle='--', alpha=0.5)
            except np.linalg.LinAlgError:
                print(f"Could not solve system with noise level {noise_level}")
        
        plt.title(f"{name}\nCondition Number: {cond:.2e}")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Plot SVD ellipse
        plt.subplot(2, 3, i+4)
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(A)
        
        # Plot unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = np.cos(theta)
        circle_y = np.sin(theta)
        plt.plot(circle_x, circle_y, 'k--', alpha=0.5, label="Unit Circle")
        
        # Plot transformed circle (ellipse)
        transformed = np.zeros((100, 2))
        for j in range(100):
            point = np.array([circle_x[j], circle_y[j]])
            transformed[j] = A @ point
        
        plt.plot(transformed[:, 0], transformed[:, 1], 'b-', label="Transformed Circle")
        
        # Plot semi-major and semi-minor axes
        plt.arrow(0, 0, S[0]*Vt[0, 0], S[0]*Vt[0, 1], color='red', head_width=0.1, 
                  head_length=0.1, linewidth=2, label="Major Axis")
        plt.arrow(0, 0, S[1]*Vt[1, 0], S[1]*Vt[1, 1], color='green', head_width=0.1, 
                  head_length=0.1, linewidth=2, label="Minor Axis")
        
        plt.axhline(y=0, color='k', alpha=0.3)
        plt.axvline(x=0, color='k', alpha=0.3)
        plt.title(f"SVD Visualization\nσ₁={S[0]:.4f}, σ₂={S[1]:.4f}")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

# Visualize the effect of condition number in 2D
visualize_condition_number_2d()

# %% [markdown]
# ## 4. Graph Analysis with Adjacency Matrices
# 
# Graphs can be represented using adjacency matrices, where matrix properties provide insights into the graph structure and connectivity. Let's explore how matrix properties can be used for graph analysis.

# %%
def create_graph_examples():
    """Create and analyze different types of graphs."""
    # Create different types of graphs
    graph_types = [
        ("Complete Graph", nx.complete_graph(10)),
        ("Path Graph", nx.path_graph(10)),
        ("Star Graph", nx.star_graph(9)),
        ("Cycle Graph", nx.cycle_graph(10)),
        ("Random Graph", nx.gnp_random_graph(10, 0.3, seed=42))
    ]
    
    # Convert to adjacency matrices
    adjacency_matrices = []
    for name, G in graph_types:
        A = nx.to_numpy_array(G)
        adjacency_matrices.append((name, G, A))
    
    return adjacency_matrices

# Create graph examples
graph_examples = create_graph_examples()

# Visualize the graphs and their adjacency matrices
plt.figure(figsize=(15, 12))

for i, (name, G, A) in enumerate(graph_examples):
    # Plot the graph
    plt.subplot(len(graph_examples), 2, 2*i+1)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, 
            font_size=10, font_weight='bold', width=1.5, edge_color='gray')
    plt.title(f"{name} Graph")
    
    # Plot the adjacency matrix
    plt.subplot(len(graph_examples), 2, 2*i+2)
    plt.imshow(A, cmap='Blues', interpolation='none')
    plt.colorbar(label="Connection")
    plt.title(f"{name} Adjacency Matrix")
    
    # Add grid lines
    plt.grid(False)
    for j in range(A.shape[0] + 1):
        plt.axhline(y=j-0.5, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=j-0.5, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Eigenvalues and Graph Properties
# 
# The eigenvalues of the adjacency matrix, or related matrices like the Laplacian matrix, provide important information about the graph structure. Let's explore the relationship between eigenvalues and graph properties.

# %%
def analyze_graph_spectra(graph_examples):
    """Analyze the eigenvalue spectra of graph adjacency and Laplacian matrices."""
    spectra = []
    
    for name, G, A in graph_examples:
        # Adjacency matrix eigenvalues
        eigvals_A = np.linalg.eigvals(A)
        eigvals_A = np.sort(eigvals_A)[::-1]  # Sort in descending order
        
        # Laplacian matrix
        L = nx.laplacian_matrix(G).toarray()
        
        # Laplacian eigenvalues
        eigvals_L = np.linalg.eigvals(L)
        eigvals_L = np.sort(eigvals_L)  # Sort in ascending order
        
        # Compute graph metrics
        avg_degree = np.mean([d for n, d in G.degree()])
        n_components = nx.number_connected_components(G)
        diameter = max(nx.diameter(C) for C in (G.subgraph(c) for c in nx.connected_components(G)))
        
        spectra.append((name, eigvals_A, eigvals_L, avg_degree, n_components, diameter))
    
    return spectra

# Analyze graph spectra
graph_spectra = analyze_graph_spectra(graph_examples)

# Visualize the eigenvalue spectra
plt.figure(figsize=(15, 10))

for i, (name, eigvals_A, eigvals_L, avg_degree, n_components, diameter) in enumerate(graph_spectra):
    # Plot adjacency matrix eigenvalues
    plt.subplot(len(graph_spectra), 2, 2*i+1)
    plt.stem(range(len(eigvals_A)), eigvals_A.real, markerfmt='o', linefmt='b-', basefmt='r-')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.title(f"{name} Adjacency Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3)
    
    # Annotate with key eigenvalues and their interpretations
    plt.annotate(f"Largest eigenvalue: {eigvals_A[0]:.2f}", xy=(0.02, 0.9), xycoords='axes fraction',
                 fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.annotate(f"Avg degree: {avg_degree:.2f}\nComponents: {n_components}\nDiameter: {diameter}",
                 xy=(0.02, 0.75), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot Laplacian matrix eigenvalues
    plt.subplot(len(graph_spectra), 2, 2*i+2)
    plt.stem(range(len(eigvals_L)), eigvals_L.real, markerfmt='o', linefmt='g-', basefmt='r-')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.title(f"{name} Laplacian Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True, alpha=0.3)
    
    # Annotate with key eigenvalues and their interpretations
    plt.annotate(f"Second eigenvalue: {eigvals_L[1]:.2f}\n(algebraic connectivity)",
                 xy=(0.02, 0.9), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.annotate(f"Zero eigenvalues: {np.sum(np.abs(eigvals_L) < 1e-10)}\n(= number of connected components)",
                 xy=(0.02, 0.75), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Print some key insights
print("Graph Spectral Analysis:")
print("-" * 80)
print(f"{'Graph Type':<20} {'Largest Adj. Eigval':<20} {'Algebraic Connectivity':<25} {'Spectral Gap':<15}")
print("-" * 80)

for name, eigvals_A, eigvals_L, avg_degree, n_components, diameter in graph_spectra:
    # Largest adjacency eigenvalue (related to average degree)
    largest_adj = eigvals_A[0].real
    
    # Algebraic connectivity (second smallest Laplacian eigenvalue)
    alg_connectivity = eigvals_L[1].real
    
    # Spectral gap (difference between largest and second largest adjacency eigenvalues)
    spectral_gap = eigvals_A[0].real - eigvals_A[1].real
    
    print(f"{name:<20} {largest_adj:<20.4f} {alg_connectivity:<25.4f} {spectral_gap:<15.4f}")

# %% [markdown]
# ### Random Walk and Markov Chains on Graphs
# 
# The adjacency matrix can be normalized to create a transition matrix for a random walk on the graph. Let's explore how this relates to the stationary distribution and mixing properties of the Markov chain.

# %%
def analyze_random_walks(graph_examples):
    """Analyze random walks on graphs using transition matrices."""
    results = []
    
    for name, G, A in graph_examples:
        # Calculate degree matrix
        degrees = np.sum(A, axis=1)
        D_inv = np.diag(1.0 / degrees)
        
        # Calculate transition matrix P = D^(-1)A
        P = D_inv @ A
        
        # Calculate eigenvalues of P
        eigvals_P = np.linalg.eigvals(P)
        eigvals_P = np.sort(np.abs(eigvals_P))[::-1]  # Sort by absolute value
        
        # Compute stationary distribution
        # For a simple random walk, it's proportional to node degrees
        stationary = degrees / np.sum(degrees)
        
        # Estimate mixing time (related to second largest eigenvalue)
        second_eigval = eigvals_P[1]
        mixing_time_estimate = -1.0 / np.log(second_eigval) if second_eigval < 1.0 else np.inf
        
        results.append((name, P, eigvals_P, stationary, mixing_time_estimate))
    
    return results

# Analyze random walks
random_walk_results = analyze_random_walks(graph_examples)

# Visualize transition matrices and eigenvalues
plt.figure(figsize=(15, 15))

for i, (name, P, eigvals_P, stationary, mixing_time) in enumerate(random_walk_results):
    # Plot transition matrix
    plt.subplot(len(random_walk_results), 2, 2*i+1)
    plt.imshow(P, cmap='Blues', interpolation='none')
    plt.colorbar(label="Transition Probability")
    plt.title(f"{name} Transition Matrix")
    
    # Add grid lines
    plt.grid(False)
    for j in range(P.shape[0] + 1):
        plt.axhline(y=j-0.5, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=j-0.5, color='gray', linestyle='-', alpha=0.3)
    
    # Plot eigenvalues of transition matrix
    plt.subplot(len(random_walk_results), 2, 2*i+2)
    plt.stem(range(len(eigvals_P)), eigvals_P, markerfmt='o', linefmt='m-', basefmt='r-')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
    plt.title(f"{name} Transition Matrix Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Absolute Eigenvalue")
    plt.grid(True, alpha=0.3)
    
    # Annotate with key values
    plt.annotate(f"Second largest eigenvalue: {eigvals_P[1]:.4f}\nEstimated mixing time: {mixing_time:.2f}",
                 xy=(0.02, 0.9), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.show()

# Print key insights
print("\nRandom Walk Analysis:")
print("-" * 80)
print(f"{'Graph Type':<20} {'Second Largest Eigval':<25} {'Mixing Time Estimate':<20} {'Spectral Gap':<15}")
print("-" * 80)

for name, P, eigvals_P, stationary, mixing_time in random_walk_results:
    # Second largest eigenvalue (related to mixing time)
    second_eigval = eigvals_P[1]
    
    # Spectral gap for transition matrix
    spectral_gap = 1.0 - second_eigval
    
    print(f"{name:<20} {second_eigval:<25.4f} {mixing_time:<20.4f} {spectral_gap:<15.4f}")

# %% [markdown]
# ## 5. Markov Chains and Transition Matrices
# 
# Markov chains are stochastic processes where the future state depends only on the current state. The transition matrix of a Markov chain has several important properties that we can analyze using matrix properties.

# %%
def create_markov_chain_examples():
    """Create example Markov chains with transition matrices."""
    examples = []
    
    # Example 1: Weather model (Sunny, Cloudy, Rainy)
    weather = np.array([
        [0.7, 0.2, 0.1],  # Sunny -> Sunny, Cloudy, Rainy
        [0.3, 0.4, 0.3],  # Cloudy -> Sunny, Cloudy, Rainy
        [0.2, 0.4, 0.4]   # Rainy -> Sunny, Cloudy, Rainy
    ])
    examples.append(("Weather", weather, ["Sunny", "Cloudy", "Rainy"]))
    
    # Example 2: Population migration between 4 cities
    migration = np.array([
        [0.7, 0.1, 0.1, 0.1],  # City A -> A, B, C, D
        [0.2, 0.6, 0.1, 0.1],  # City B -> A, B, C, D
        [0.1, 0.1, 0.7, 0.1],  # City C -> A, B, C, D
        [0.1, 0.1, 0.1, 0.7]   # City D -> A, B, C, D
    ])
    examples.append(("Migration", migration, ["City A", "City B", "City C", "City D"]))
    
    # Example 3: Social class mobility
    mobility = np.array([
        [0.7, 0.2, 0.1],  # Low -> Low, Medium, High
        [0.3, 0.5, 0.2],  # Medium -> Low, Medium, High
        [0.1, 0.3, 0.6]   # High -> Low, Medium, High
    ])
    examples.append(("Social Mobility", mobility, ["Low", "Medium", "High"]))
    
    # Example 4: Queuing system (0, 1, 2, 3 customers in queue)
    queue = np.array([
        [0.3, 0.7, 0.0, 0.0],  # 0 -> 0, 1, 2, 3
        [0.4, 0.3, 0.3, 0.0],  # 1 -> 0, 1, 2, 3
        [0.0, 0.5, 0.2, 0.3],  # 2 -> 0, 1, 2, 3
        [0.0, 0.0, 0.6, 0.4]   # 3 -> 0, 1, 2, 3
    ])
    examples.append(("Queue", queue, ["0", "1", "2", "3"]))
    
    return examples

# Create Markov chain examples
markov_examples = create_markov_chain_examples()

# Visualize the transition matrices
plt.figure(figsize=(12, 10))

for i, (name, P, states) in enumerate(markov_examples):
    plt.subplot(2, 2, i+1)
    plt.imshow(P, cmap='Blues', interpolation='none')
    plt.colorbar(label="Transition Probability")
    plt.title(f"{name} Transition Matrix")
    
    # Add state labels
    plt.xticks(range(len(states)), states)
    plt.yticks(range(len(states)), states)
    
    # Add grid lines
    plt.grid(False)
    for j in range(P.shape[0] + 1):
        plt.axhline(y=j-0.5, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=j-0.5, color='gray', linestyle='-', alpha=0.3)
    
    # Annotate probabilities
    for ii in range(P.shape[0]):
        for jj in range(P.shape[1]):
            plt.text(jj, ii, f"{P[ii, jj]:.1f}", ha='center', va='center', 
                     color='white' if P[ii, jj] > 0.5 else 'black')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Analyzing Markov Chain Properties
# 
# Let's analyze the properties of our Markov chains, including:
# 
# - Stationary distribution
# - Eigenvalue spectrum
# - Mixing time
# - Classification of states

# %%
def analyze_markov_chains(markov_examples):
    """Analyze properties of Markov chain transition matrices."""
    results = []
    
    for name, P, states in markov_examples:
        # Calculate eigenvalues and eigenvectors
        eigvals, eigvecs = np.linalg.eig(P.T)  # Transpose for left eigenvectors
        
        # Sort eigenvalues and eigenvectors
        idx = np.argsort(np.abs(eigvals))[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Find stationary distribution (corresponds to eigenvalue 1)
        stationary = np.real(eigvecs[:, 0])
        stationary = stationary / np.sum(stationary)  # Normalize
        
        # Estimate mixing time
        second_eigval = np.abs(eigvals[1])
        mixing_time = -1.0 / np.log(second_eigval) if second_eigval < 1.0 else np.inf
        
        # Check for periodicity and irreducibility
        is_irreducible = np.all(np.linalg.matrix_power(P, len(states)-1) > 0)
        
        # Check if any eigenvalues have magnitude 1 but are not 1
        has_periodicity = np.any(np.abs(np.abs(eigvals) - 1.0) < 1e-10 & np.abs(eigvals - 1.0) > 1e-10)
        
        results.append((name, eigvals, stationary, mixing_time, 
                        is_irreducible, has_periodicity, states))
    
    return results

# Analyze Markov chains
markov_results = analyze_markov_chains(markov_examples)

# Visualize the results
plt.figure(figsize=(15, 15))

for i, (name, eigvals, stationary, mixing_time, is_irreducible, has_periodicity, states) in enumerate(markov_results):
    # Plot eigenvalues in the complex plane
    plt.subplot(len(markov_results), 2, 2*i+1)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
    
    # Plot eigenvalues
    plt.scatter(eigvals.real, eigvals.imag, color='blue', alpha=0.7)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Highlight eigenvalue 1
    plt.scatter([1], [0], color='red', s=100, edgecolor='black', zorder=3)
    
    plt.title(f"{name} Eigenvalues")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Annotate with key properties
    chain_type = "Irreducible" if is_irreducible else "Reducible"
    if has_periodicity:
        chain_type += ", Periodic"
    else:
        chain_type += ", Aperiodic"
    
    plt.annotate(f"Chain type: {chain_type}\nMixing time: {mixing_time:.2f}",
                 xy=(0.02, 0.9), xycoords='axes fraction', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Plot stationary distribution
    plt.subplot(len(markov_results), 2, 2*i+2)
    plt.bar(range(len(states)), stationary, color='skyblue', edgecolor='blue')
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.title(f"{name} Stationary Distribution")
    plt.xticks(range(len(states)), states)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line for uniform distribution
    plt.axhline(y=1.0/len(states), color='red', linestyle='--', label="Uniform")
    plt.legend()

plt.tight_layout()
plt.show()

# Print key insights
print("\nMarkov Chain Analysis:")
print("-" * 100)
print(f"{'Chain Type':<15} {'Stationary Distribution':<40} {'Mixing Time':<15} {'2nd Largest Eigval':<15} {'Type':<15}")
print("-" * 100)

for name, eigvals, stationary, mixing_time, is_irreducible, has_periodicity, states in markov_results:
    stationary_str = ", ".join([f"{s:.3f}" for s in stationary])
    
    chain_type = "Irreducible" if is_irreducible else "Reducible"
    if has_periodicity:
        chain_type += ", Periodic"
    else:
        chain_type += ", Aperiodic"
    
    print(f"{name:<15} {stationary_str:<40} {mixing_time:<15.2f} {np.abs(eigvals[1]):<15.4f} {chain_type:<15}")

# %% [markdown]
# ### Simulating Markov Chain Evolution
# 
# Let's simulate the evolution of state probabilities over time for our Markov chains.

# %%
def simulate_markov_evolution(markov_examples, n_steps=20):
    """Simulate the evolution of state probabilities over time."""
    plt.figure(figsize=(15, 10))
    
    for i, (name, P, states) in enumerate(markov_examples):
        plt.subplot(2, 2, i+1)
        
        # Create different initial distributions
        n_states = len(states)
        distributions = [
            np.eye(n_states)[0],  # Start in state 0
            np.eye(n_states)[-1],  # Start in last state
            np.ones(n_states) / n_states,  # Uniform distribution
            np.random.dirichlet(np.ones(n_states))  # Random distribution
        ]
        
        labels = ["Start in first state", "Start in last state", 
                  "Uniform distribution", "Random distribution"]
        
        # Simulate evolution for each initial distribution
        for j, (dist, label) in enumerate(zip(distributions, labels)):
            # Initialize state vector
            x = dist
            
            # Record evolution
            evolution = np.zeros((n_steps, n_states))
            evolution[0] = x
            
            # Evolve the state
            for t in range(1, n_steps):
                x = x @ P
                evolution[t] = x
            
            # Plot the evolution of each state probability
            for k in range(n_states):
                plt.plot(range(n_steps), evolution[:, k], 
                         color=f'C{j}', linestyle=['-', '--', ':', '-.'][j], 
                         alpha=0.7 if j == 0 else 0.5,
                         label=f"{label}, State {states[k]}" if k == 0 else "")
        
        plt.title(f"{name} Markov Chain Evolution")
        plt.xlabel("Time Step")
        plt.ylabel("State Probability")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True, alpha=0.3)
        
        # Calculate and plot stationary distribution
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = np.abs(eigvals - 1.0) < 1e-10
        stationary = np.real(eigvecs[:, idx][:, 0])
        stationary = stationary / np.sum(stationary)
        
        for k in range(n_states):
            plt.axhline(y=stationary[k], color=f'C{k}', linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    plt.show()

# Simulate Markov chain evolution
simulate_markov_evolution(markov_examples[:4])  # Use all examples

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored various applications of matrix properties:
# 
# 1. **Image Processing and Compression**:
#    - Using SVD for image compression
#    - Understanding how singular values relate to image information
# 
# 2. **Principal Component Analysis (PCA)**:
#    - Using eigendecomposition of the covariance matrix to find principal components
#    - Visualizing data in reduced dimensions
# 
# 3. **Linear Systems and Condition Number**:
#    - Understanding how condition number affects numerical stability
#    - Visualizing the effect of noise on solution accuracy
# 
# 4. **Graph Analysis with Adjacency Matrices**:
#    - Using eigenvalues to understand graph properties
#    - Analyzing random walks on graphs
# 
# 5. **Markov Chains and Transition Matrices**:
#    - Finding stationary distributions
#    - Analyzing mixing times and convergence properties
# 
# These applications demonstrate the power and versatility of matrix properties in solving real-world problems across various domains.