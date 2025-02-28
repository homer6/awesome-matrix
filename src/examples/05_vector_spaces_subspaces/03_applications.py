# %% [markdown]
# # Vector Spaces: Applications and Examples
# 
# In this notebook, we'll explore practical applications of vector spaces, focusing on coordinate transformations, image processing, and data representation. These examples will demonstrate how the abstract concepts of basis, orthogonality, and projection can be applied to solve real-world problems.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
import requests
from io import BytesIO
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display

# For better looking plots
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Application 1: Coordinate Transformations in Computer Graphics
# 
# In computer graphics, coordinate transformations are used to move, rotate, and scale objects. These transformations can be represented as changes of basis in vector spaces.
# 
# Let's visualize how a simple 2D shape is transformed under different coordinate transformations.

# %%
def plot_coordinate_transformation(points, transformation_matrix, title):
    """Plot a shape before and after a coordinate transformation."""
    # Convert points to numpy array if needed
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    # Apply transformation to points
    transformed_points = points @ transformation_matrix.T
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Original shape
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
    plt.plot(np.append(points[:, 0], points[0, 0]), 
             np.append(points[:, 1], points[0, 1]), 'b-')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title('Original Shape')
    
    # Draw coordinate axes
    max_val = max(np.max(np.abs(points)), 5)
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Transformed shape
    plt.subplot(1, 2, 2)
    plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color='red', s=50)
    plt.plot(np.append(transformed_points[:, 0], transformed_points[0, 0]), 
             np.append(transformed_points[:, 1], transformed_points[0, 1]), 'r-')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.title(f'After {title}')
    
    # Draw coordinate axes
    max_val = max(np.max(np.abs(transformed_points)), 5)
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create a simple shape (square)
square_points = np.array([
    [1, 1],
    [1, -1],
    [-1, -1],
    [-1, 1]
])

# Define transformation matrices
# 1. Rotation by 45 degrees
theta = np.pi/4
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# 2. Scaling
scaling_matrix = np.array([
    [2, 0],
    [0, 0.5]
])

# 3. Shearing
shear_matrix = np.array([
    [1, 0.5],
    [0, 1]
])

# 4. Reflection about y-axis
reflection_matrix = np.array([
    [-1, 0],
    [0, 1]
])

# Plot transformations
plot_coordinate_transformation(square_points, rotation_matrix, "Rotation (45°)")
plot_coordinate_transformation(square_points, scaling_matrix, "Scaling (2x horizontally, 0.5x vertically)")
plot_coordinate_transformation(square_points, shear_matrix, "Shearing")
plot_coordinate_transformation(square_points, reflection_matrix, "Reflection about y-axis")

# %% [markdown]
# Each transformation can be interpreted as a change of basis in the 2D vector space. For example:
# 
# 1. **Rotation**: Changes the basis to a rotated coordinate system
# 2. **Scaling**: Changes the basis to one where the unit vectors have different lengths
# 3. **Shearing**: Changes the basis to one where the basis vectors are no longer orthogonal
# 4. **Reflection**: Changes the basis to one where one basis vector points in the opposite direction
# 
# ## Application 2: Change of Basis with Standard and Rotated Axes
# 
# Now let's visualize a specific example of a change of basis by showing how points are represented in two different coordinate systems.

# %%
def plot_change_of_basis_points(points, standard_basis, rotated_basis):
    """Plot points in both standard and rotated coordinate systems."""
    # Convert to numpy
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    standard_basis = [b.numpy() if isinstance(b, torch.Tensor) else b for b in standard_basis]
    rotated_basis = [b.numpy() if isinstance(b, torch.Tensor) else b for b in rotated_basis]
    
    # Create matrices where columns are basis vectors
    B_std = np.column_stack(standard_basis)
    B_rot = np.column_stack(rotated_basis)
    
    # Change of basis matrix from standard to rotated
    P_std_to_rot = np.linalg.inv(B_rot) @ B_std
    
    # Convert points to rotated basis coordinates
    points_in_rotated = points @ P_std_to_rot.T
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot in standard basis
    plt.subplot(1, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=50)
    
    # Draw standard basis vectors
    plt.arrow(0, 0, standard_basis[0][0], standard_basis[0][1], 
              head_width=0.1, head_length=0.1, fc='black', ec='black', width=0.02)
    plt.arrow(0, 0, standard_basis[1][0], standard_basis[1][1], 
              head_width=0.1, head_length=0.1, fc='black', ec='black', width=0.02)
    
    # Draw grid lines for standard basis
    max_val = max(np.max(np.abs(points)), 5)
    for i in range(-5, 6):
        plt.axhline(y=i, color='gray', linestyle='-', alpha=0.2)
        plt.axvline(x=i, color='gray', linestyle='-', alpha=0.2)
    
    plt.grid(False)
    plt.axis('equal')
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.title('Points in Standard Basis')
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Plot in rotated basis
    plt.subplot(1, 2, 2)
    plt.scatter(points_in_rotated[:, 0], points_in_rotated[:, 1], color='red', s=50)
    
    # Create rotated grid
    grid_lines = 5
    rot_x = rotated_basis[0]
    rot_y = rotated_basis[1]
    
    # Draw rotated basis vectors
    plt.arrow(0, 0, rot_x[0], rot_x[1], 
              head_width=0.1, head_length=0.1, fc='black', ec='black', width=0.02)
    plt.arrow(0, 0, rot_y[0], rot_y[1], 
              head_width=0.1, head_length=0.1, fc='black', ec='black', width=0.02)
    
    # Draw grid lines for rotated basis
    for i in range(-5, 6):
        plt.plot([i*rot_x[0], i*rot_x[0] + 10*rot_y[0]], 
                 [i*rot_x[1], i*rot_x[1] + 10*rot_y[1]], 
                 'gray', alpha=0.2)
        plt.plot([i*rot_y[0], i*rot_y[0] + 10*rot_x[0]], 
                 [i*rot_y[1], i*rot_y[1] + 10*rot_x[1]], 
                 'gray', alpha=0.2)
    
    plt.grid(False)
    plt.axis('equal')
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.title('Points in Rotated Basis')
    plt.xlabel("x'")
    plt.ylabel("y'")
    
    plt.tight_layout()
    plt.show()
    
    # Print the coordinates in both bases
    print("Points in standard basis:")
    print(points)
    print("\nPoints in rotated basis:")
    print(points_in_rotated)

# Create some points
points = np.array([
    [3, 2],
    [1, 4],
    [4, -1],
    [-2, -3],
    [0, 0]
])

# Standard basis
std_basis = [np.array([1, 0]), np.array([0, 1])]

# Rotated basis (30 degrees)
theta = np.pi/6  # 30 degrees
rot_basis = [
    np.array([np.cos(theta), np.sin(theta)]),
    np.array([-np.sin(theta), np.cos(theta)])
]

plot_change_of_basis_points(points, std_basis, rot_basis)

# %% [markdown]
# This visualization shows how the same points can be represented in different coordinate systems. The points maintain their geometric relationships, but their coordinates change according to the basis.
# 
# ## Application 3: Image Processing with Projection
# 
# Projection is widely used in image processing and compression. One common application is noise reduction by projecting an image onto a lower-dimensional subspace.
# 
# Let's visualize how projection works by adding noise to an image and then removing it.

# %%
def load_and_process_image():
    """Load and process a sample image."""
    # Download a grayscale image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/da/Claude_Shannon_MF.jpg/800px-Claude_Shannon_MF.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
    
    # Resize for faster processing
    img = img.resize((200, 200))
    
    # Convert to numpy array
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    
    return img_array

def add_noise(img_array, noise_level=0.1):
    """Add random noise to an image."""
    noise = np.random.normal(0, noise_level, img_array.shape)
    noisy_img = img_array + noise
    return np.clip(noisy_img, 0, 1)  # Clip to valid range

def project_onto_subspace(img_array, n_components):
    """Project image onto a subspace using PCA."""
    # Flatten the image
    h, w = img_array.shape
    flat_img = img_array.reshape(1, -1)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    pca.fit(flat_img)
    
    # Project onto principal components and reconstruct
    projected = pca.transform(flat_img)
    reconstructed = pca.inverse_transform(projected)
    
    # Reshape back to image
    return reconstructed.reshape(h, w)

def plot_image_projection(img_array, noisy_img, projected_img):
    """Plot original, noisy, and projected images."""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(projected_img, cmap='gray')
    plt.title('Projected Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load and process image
try:
    img_array = load_and_process_image()
    
    # Add noise
    noisy_img = add_noise(img_array, noise_level=0.1)
    
    # Project onto subspace
    projected_img = project_onto_subspace(noisy_img, n_components=50)
    
    # Plot
    plot_image_projection(img_array, noisy_img, projected_img)
except Exception as e:
    print(f"Error loading or processing image: {e}")
    # Use a random image instead
    img_array = np.random.rand(200, 200)
    noisy_img = add_noise(img_array, noise_level=0.1)
    projected_img = project_onto_subspace(noisy_img, n_components=50)
    plot_image_projection(img_array, noisy_img, projected_img)

# %% [markdown]
# This example demonstrates how projection can be used to reduce noise in an image. The noisy image is projected onto a lower-dimensional subspace defined by the principal components, which captures the most important features while discarding the noise.
# 
# ## Application 4: Principal Component Analysis (PCA)
# 
# PCA is a technique that uses eigenvalues and eigenvectors to find a new basis where the data has maximum variance along each basis vector. This is a practical application of the concept of change of basis.
# 
# Let's visualize how PCA works with a simple 2D dataset.

# %%
def generate_correlated_data(n_samples=100, angle=np.pi/6, scale=[3, 1]):
    """Generate correlated 2D data."""
    # Generate random data
    X = np.random.randn(n_samples, 2)
    
    # Scale to create different variances
    X = X * np.array(scale)
    
    # Rotate to create correlation
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    X = X @ rotation_matrix.T
    
    return X

def plot_pca_result(X, pca, n_components=2):
    """Plot PCA result with data points and principal components."""
    plt.figure(figsize=(12, 6))
    
    # Plot original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
    
    # Plot mean point
    mean = pca.mean_
    plt.scatter(mean[0], mean[1], color='red', s=100, marker='x')
    
    # Plot principal components
    for i in range(n_components):
        component = pca.components_[i]
        variance = pca.explained_variance_[i]
        plt.arrow(mean[0], mean[1], component[0] * variance, component[1] * variance,
                 head_width=0.2, head_length=0.3, fc='k', ec='k', width=0.05)
        plt.text(mean[0] + component[0] * variance * 1.1, 
                mean[1] + component[1] * variance * 1.1, 
                f"PC{i+1}", fontsize=12)
    
    # Draw grid lines
    max_val = np.max(np.abs(X)) * 1.2
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.title('Original Data with Principal Components')
    
    # Plot transformed data
    plt.subplot(1, 2, 2)
    X_transformed = pca.transform(X)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.7)
    
    # Draw new axes
    max_val = np.max(np.abs(X_transformed)) * 1.2
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.gca().set_aspect('equal')
    plt.title('Data in Principal Component Space')
    
    plt.tight_layout()
    plt.show()
    
    # Print explained variance
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Explained variance:", pca.explained_variance_)

# Generate data
X = generate_correlated_data(n_samples=200, angle=np.pi/6, scale=[3, 1])

# Perform PCA
pca = PCA(n_components=2)
pca.fit(X)

# Plot results
plot_pca_result(X, pca)

# %% [markdown]
# PCA finds a new basis where:
# 
# 1. The first principal component (PC1) points in the direction of maximum variance
# 2. The second principal component (PC2) is orthogonal to PC1 and points in the direction of maximum remaining variance
# 
# This is a change of basis that aligns the coordinate system with the natural directions of variability in the data.
# 
# ## Application 5: Face Recognition with Eigenfaces
# 
# Eigenfaces is a technique for face recognition that uses PCA to find a basis for the "face space." This is a direct application of vector spaces, subspaces, and basis vectors to a real-world problem.
# 
# Let's explore this technique with a dataset of faces.

# %%
def plot_eigenfaces(pca, n_components=10):
    """Plot the eigenfaces (principal components)."""
    plt.figure(figsize=(12, 4))
    
    for i in range(min(n_components, pca.components_.shape[0])):
        plt.subplot(2, 5, i + 1)
        # Reshape component to image shape
        eigenvector = pca.components_[i].reshape(64, 64)
        # Scale for better visualization
        eigenvector = (eigenvector - eigenvector.min()) / (eigenvector.max() - eigenvector.min())
        plt.imshow(eigenvector, cmap='viridis')
        plt.title(f"Eigenface {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_face_reconstruction(X, pca, face_idx=0, n_components_list=[1, 5, 10, 20, 50, 100, 200]):
    """Plot a face reconstruction using different numbers of eigenfaces."""
    plt.figure(figsize=(15, 4))
    
    # Original face
    plt.subplot(1, len(n_components_list) + 1, 1)
    plt.imshow(X[face_idx].reshape(64, 64), cmap='gray')
    plt.title("Original")
    plt.axis('off')
    
    # Reconstructions with increasing number of components
    for i, n in enumerate(n_components_list):
        # Project to PCA space and back
        X_projected = pca.transform(X[face_idx:face_idx+1, :])[:, :n]
        X_reconstructed = X_projected @ pca.components_[:n, :]
        X_reconstructed += pca.mean_
        
        # Plot
        plt.subplot(1, len(n_components_list) + 1, i + 2)
        plt.imshow(X_reconstructed.reshape(64, 64), cmap='gray')
        plt.title(f"{n} Components")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load face dataset
try:
    faces = fetch_olivetti_faces(shuffle=True)
    X = faces.data
    y = faces.target
    
    # Perform PCA
    pca = PCA().fit(X)
    
    # Plot eigenfaces
    plot_eigenfaces(pca)
    
    # Plot face reconstruction
    plot_face_reconstruction(X, pca)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.title('Explained Variance vs. Components')
    
    plt.subplot(1, 2, 2)
    plt.semilogy(pca.explained_variance_)
    plt.xlabel('Component Index')
    plt.ylabel('Eigenvalue (log scale)')
    plt.grid(True)
    plt.title('Eigenvalue Spectrum')
    
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"Error loading or processing faces dataset: {e}")
    print("Skipping eigenfaces example.")

# %% [markdown]
# The eigenfaces technique demonstrates how a high-dimensional space (face images) can be represented using a lower-dimensional subspace (spanned by eigenfaces). Each face can be represented as a linear combination of these eigenfaces, making it a direct application of the basis concept in vector spaces.
# 
# ## Application 6: Projection for Least Squares Fitting
# 
# Projection is the mathematical basis of least squares fitting, which is widely used in regression analysis.
# 
# Let's visualize how projection can be used to find the best-fit line for a set of points.

# %%
def least_squares_fit(X, y):
    """Compute least squares fit."""
    # Add column of ones for intercept
    X_with_ones = np.column_stack([np.ones(len(X)), X])
    
    # Compute coefficients: b = (X^T X)^-1 X^T y
    # This is the projection of y onto the column space of X
    b = np.linalg.inv(X_with_ones.T @ X_with_ones) @ X_with_ones.T @ y
    
    return b

def plot_least_squares(X, y, b):
    """Plot data points and the least squares fit line."""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X, y, alpha=0.7, label='Data points')
    
    # Plot fit line
    x_range = np.linspace(min(X), max(X), 100)
    y_fit = b[0] + b[1] * x_range
    plt.plot(x_range, y_fit, 'r-', label=f'y = {b[0]:.2f} + {b[1]:.2f}x')
    
    # Plot projection lines
    X_with_ones = np.column_stack([np.ones(len(X)), X])
    y_proj = X_with_ones @ b
    
    for i in range(len(X)):
        plt.plot([X[i], X[i]], [y[i], y_proj[i]], 'k--', alpha=0.3)
    
    # Add labels and legend
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Least Squares Fit as Projection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print coefficients and error
    error = np.mean((y - y_proj)**2)
    print(f"Coefficients: Intercept = {b[0]:.4f}, Slope = {b[1]:.4f}")
    print(f"Mean Squared Error: {error:.4f}")

# Generate data with noise
np.random.seed(42)
X = np.linspace(0, 10, 20)
y = 2 * X + 5 + np.random.normal(0, 2, size=len(X))

# Compute least squares fit
b = least_squares_fit(X, y)

# Plot
plot_least_squares(X, y, b)

# %% [markdown]
# In this example, the least squares fit is computed by projecting the observed y values onto the column space of the design matrix X. The projection minimizes the sum of squared errors, which is geometrically the shortest distance from the data points to the fitting line.
# 
# ## Conclusion
# 
# In this notebook, we've explored several practical applications of vector spaces:
# 
# 1. **Coordinate Transformations**: We visualized how shapes can be transformed by changing the basis of the vector space.
# 
# 2. **Change of Basis**: We demonstrated how the same points can be represented in different coordinate systems.
# 
# 3. **Image Processing**: We used projection to reduce noise in an image by projecting onto a lower-dimensional subspace.
# 
# 4. **Principal Component Analysis (PCA)**: We visualized how PCA finds a new basis that maximizes variance along each basis vector.
# 
# 5. **Face Recognition with Eigenfaces**: We applied PCA to face images to find a basis for the "face space."
# 
# 6. **Least Squares Fitting**: We showed how projection can be used to find the best-fit line for a set of points.
# 
# These applications demonstrate how the abstract concepts of vector spaces, basis, orthogonality, and projection can be applied to solve real-world problems in diverse fields such as computer graphics, image processing, data analysis, and machine learning.