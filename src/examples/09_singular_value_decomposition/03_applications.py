# %% [markdown]
# # Singular Value Decomposition: Applications
# 
# The Singular Value Decomposition (SVD) is a powerful matrix factorization with applications across numerous fields. This notebook explores practical applications of SVD in various domains, demonstrating its versatility and utility.
# 
# We'll explore the following applications:
# 
# 1. **Image Compression**
# 2. **Principal Component Analysis (PCA)**
# 3. **Latent Semantic Analysis (LSA)**
# 4. **Recommendation Systems**
# 5. **Signal Processing and Denoising**
# 6. **Pseudoinverse and Least Squares Solutions**
# 
# Each application showcases how the mathematical properties of SVD can be leveraged to solve real-world problems.

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import scipy.linalg
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_olivetti_faces, make_blobs, load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from scipy import signal

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
# ## 1. Image Compression
# 
# SVD provides an effective way to compress images by representing them with a lower-rank approximation. This is particularly useful because the singular values typically decay rapidly, meaning that we can capture most of the image's information with just a few components.
# 
# Let's demonstrate image compression using SVD:

# %%
def load_image(url=None, size=(512, 512)):
    """Load an image from a URL or use a local file."""
    try:
        if url:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
        else:
            # Use a scikit-learn sample image
            faces = fetch_olivetti_faces()
            face = faces.images[0]
            img = Image.fromarray((face * 255).astype(np.uint8))
        
        # Convert to grayscale and resize
        img = img.convert('L').resize(size)
        return np.array(img)
    except:
        # Create a synthetic image if loading fails
        print("Failed to load image, creating a synthetic one")
        x, y = np.meshgrid(np.linspace(-3, 3, size[0]), np.linspace(-3, 3, size[1]))
        img = np.exp(-(x**2 + y**2) / 2) * 255
        return img.astype(np.uint8)

def compress_image_with_svd(image, k_values=None):
    """
    Compress an image using SVD with different numbers of singular values.
    
    Args:
        image: Input image (2D numpy array)
        k_values: List of k values (numbers of singular components) to use
        
    Returns:
        compressed_images: Dictionary mapping k values to compressed images
        U, S, V: SVD components of the image
    """
    # Compute SVD of the image
    U, S, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Default k values if not provided
    if k_values is None:
        max_k = min(image.shape)
        k_values = [1, 5, 10, 20, 50, 100, max_k]
        k_values = [k for k in k_values if k <= max_k]
    
    # Reconstruct the image with different numbers of singular values
    compressed_images = {}
    for k in k_values:
        # Truncate the SVD
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct
        compressed = U_k @ np.diag(S_k) @ Vt_k
        compressed_images[k] = np.clip(compressed, 0, 255).astype(np.uint8)
    
    return compressed_images, U, S, Vt

def calculate_compression_ratio(image, k):
    """Calculate the compression ratio achieved with k singular values."""
    m, n = image.shape
    original_size = m * n
    compressed_size = k * (m + n + 1)  # k values in each of U, S, V
    return original_size / compressed_size

def calculate_image_quality(original, compressed):
    """Calculate the PSNR (Peak Signal-to-Noise Ratio) between images."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

def demonstrate_image_compression():
    """Demonstrate image compression using SVD."""
    # Load an image
    image = load_image(size=(256, 256))
    
    # Display the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.show()
    
    # Compress the image with different numbers of singular values
    max_k = min(image.shape)
    k_values = [1, 5, 10, 20, 50, 100, int(max_k/4), int(max_k/2)]
    k_values = sorted(list(set([k for k in k_values if k <= max_k])))
    
    compressed_images, U, S, Vt = compress_image_with_svd(image, k_values)
    
    # Display the compressed images
    plt.figure(figsize=(15, 8))
    for i, k in enumerate(k_values):
        plt.subplot(2, 4, i+1)
        plt.imshow(compressed_images[k], cmap='gray')
        compression_ratio = calculate_compression_ratio(image, k)
        psnr = calculate_image_quality(image, compressed_images[k])
        plt.title(f"k={k}\nRatio: {compression_ratio:.1f}:1\nPSNR: {psnr:.1f} dB")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Plot the singular values
    plt.figure(figsize=(10, 6))
    plt.semilogy(S, 'o-')
    plt.title("Singular Values (Log Scale)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Plot cumulative energy
    energy = S**2 / np.sum(S**2)
    cumulative_energy = np.cumsum(energy)
    
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_energy, 'o-')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90%')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99%')
    plt.title("Cumulative Energy of Singular Values")
    plt.xlabel("Number of Singular Values")
    plt.ylabel("Cumulative Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Find the number of singular values needed to capture 90% and 99% of energy
    k_90 = np.sum(cumulative_energy < 0.9) + 1
    k_99 = np.sum(cumulative_energy < 0.99) + 1
    
    print(f"Number of singular values needed to capture 90% of energy: {k_90}")
    print(f"Number of singular values needed to capture 99% of energy: {k_99}")
    
    # Visualize compression ratio vs. image quality
    compression_ratios = [calculate_compression_ratio(image, k) for k in k_values]
    psnr_values = [calculate_image_quality(image, compressed_images[k]) for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(compression_ratios, psnr_values, 'o-')
    for i, k in enumerate(k_values):
        plt.annotate(f"k={k}", (compression_ratios[i], psnr_values[i]), 
                     textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title("Compression Ratio vs. Image Quality")
    plt.xlabel("Compression Ratio")
    plt.ylabel("PSNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return image, compressed_images, U, S, Vt

# Demonstrate image compression with SVD
image_orig, image_compressed, U_img, S_img, Vt_img = demonstrate_image_compression()

# %% [markdown]
# ## 2. Principal Component Analysis (PCA)
# 
# Principal Component Analysis (PCA) is a dimensionality reduction technique that can be implemented using SVD. PCA finds the directions of maximum variance in high-dimensional data and projects the data onto a lower-dimensional subspace.
# 
# The relationship between SVD and PCA is as follows:
# - For a centered data matrix $X$ (where each column has zero mean), the principal components are the right singular vectors of $X$
# - The singular values give the standard deviations of the data along each principal component
# 
# Let's demonstrate PCA using SVD:

# %%
def pca_with_svd(X, n_components=2):
    """
    Perform PCA using SVD.
    
    Args:
        X: Data matrix (samples x features)
        n_components: Number of principal components to keep
        
    Returns:
        X_transformed: Data projected onto principal components
        components: Principal components (eigenvectors)
        explained_variance: Variance explained by each component
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components are right singular vectors (rows of Vt)
    components = Vt[:n_components]
    
    # Project data onto principal components
    X_transformed = X_centered @ components.T
    
    # Variance explained is proportional to squared singular values
    total_variance = np.sum(S**2)
    explained_variance = (S**2)[:n_components] / total_variance
    
    return X_transformed, components, explained_variance

def demonstrate_pca():
    """Demonstrate PCA using SVD."""
    # Generate a 2D dataset with a clear structure
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)
    X = X @ np.array([[1, 0.7], [0.7, 1]])  # Add correlation
    
    # Perform PCA
    X_pca, components, explained_variance = pca_with_svd(X, n_components=2)
    
    # Visualize the original data and principal components
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
    
    # Plot principal components
    mean = np.mean(X, axis=0)
    for i, (comp, var) in enumerate(zip(components, explained_variance)):
        plt.arrow(mean[0], mean[1], comp[0]*3, comp[1]*3, 
                 head_width=0.3, head_length=0.5, fc=f'C{i}', ec=f'C{i}',
                 label=f"PC{i+1} ({var:.1%})")
    
    plt.title("Original Data with Principal Components")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title("Data Projected onto Principal Components")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Now demonstrate PCA on a high-dimensional dataset
    # Load the digits dataset
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target
    
    # Perform PCA
    X_digits_pca, digits_components, digits_variance = pca_with_svd(X_digits, n_components=2)
    
    # Visualize the projection
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_digits_pca[:, 0], X_digits_pca[:, 1], c=y_digits, 
                         cmap='tab10', alpha=0.7)
    plt.title("Handwritten Digits Projected onto First Two Principal Components")
    plt.xlabel(f"PC1 ({digits_variance[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({digits_variance[1]:.1%} variance)")
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, ticks=range(10), label="Digit")
    plt.show()
    
    # Visualize some of the principal components (as images)
    n_components_to_show = 10
    plt.figure(figsize=(15, 3))
    for i in range(n_components_to_show):
        plt.subplot(1, n_components_to_show, i+1)
        component = digits_components[i].reshape(8, 8)
        plt.imshow(component, cmap='viridis')
        plt.title(f"PC{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Determine the number of components needed to explain 90% of variance
    cumulative_variance = np.cumsum(digits_variance)
    n_components_90 = np.sum(cumulative_variance < 0.9) + 1
    
    # Visualize the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance, 'o-')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
    plt.axvline(x=n_components_90, color='g', linestyle='--', 
               label=f'{n_components_90} components')
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"Number of components needed to explain 90% of variance: {n_components_90}")
    print(f"Compression ratio: {X_digits.shape[1] / n_components_90:.1f}:1")
    
    return X, X_pca, components, X_digits, X_digits_pca, digits_components

# Demonstrate PCA using SVD
X_2d, X_2d_pca, pca_components, X_digits, X_digits_pca, digits_components = demonstrate_pca()

# %% [markdown]
# ## 3. Latent Semantic Analysis (LSA)
# 
# Latent Semantic Analysis (LSA) is a technique in natural language processing for analyzing relationships between documents and the terms they contain. It uses SVD to identify patterns in the relationships between the terms and concepts contained in an unstructured collection of text.
# 
# In LSA, we start with a term-document matrix where each entry represents the occurrence of a term in a document. SVD is then applied to reduce the dimensionality and uncover latent (hidden) semantic structure.
# 
# Let's demonstrate LSA on a small text corpus:

# %%
def demonstrate_lsa():
    """Demonstrate Latent Semantic Analysis using SVD."""
    # Create a small corpus of documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "The dog barks at the fox",
        "The fox is quick and brown",
        "The lazy dog sleeps all day",
        "Quick animals include foxes and rabbits",
        "Dogs are domesticated animals",
        "Some animals are lazy while others are quick"
    ]
    
    # Create a term-document matrix using CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents).toarray()
    terms = vectorizer.get_feature_names_out()
    
    # Display the term-document matrix
    df_tdm = pd.DataFrame(X, columns=terms, index=[f"Doc {i+1}" for i in range(len(documents))])
    print("Term-Document Matrix:")
    print(df_tdm)
    
    # Apply SVD for LSA
    n_components = 2  # Number of topics
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Project documents into the LSA space
    X_lsa = U[:, :n_components] @ np.diag(S[:n_components])
    
    # Get the term-topic matrix (how much each term contributes to each topic)
    term_topic = Vt[:n_components, :]
    
    # Visualize document-topic relationships
    plt.figure(figsize=(10, 8))
    plt.scatter(X_lsa[:, 0], X_lsa[:, 1])
    
    # Label each point with the document number
    for i in range(len(documents)):
        plt.annotate(f"Doc {i+1}", (X_lsa[i, 0], X_lsa[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Documents in LSA Space")
    plt.xlabel("Topic 1")
    plt.ylabel("Topic 2")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Visualize term-topic relationships
    plt.figure(figsize=(10, 8))
    plt.scatter(term_topic[0, :], term_topic[1, :])
    
    # Label each point with the term
    for i, term in enumerate(terms):
        plt.annotate(term, (term_topic[0, i], term_topic[1, i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Terms in LSA Space")
    plt.xlabel("Topic 1")
    plt.ylabel("Topic 2")
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Display the most important terms for each topic
    n_top_terms = 5
    for i in range(n_components):
        top_terms_idx = np.argsort(np.abs(term_topic[i, :]))[::-1][:n_top_terms]
        top_terms = [(terms[idx], term_topic[i, idx]) for idx in top_terms_idx]
        
        print(f"\nTop terms for Topic {i+1}:")
        for term, weight in top_terms:
            print(f"{term}: {weight:.3f}")
    
    # Calculate document similarity in the LSA space
    from sklearn.metrics.pairwise import cosine_similarity
    doc_sim = cosine_similarity(X_lsa)
    
    # Visualize document similarity
    plt.figure(figsize=(10, 8))
    sns.heatmap(doc_sim, annot=True, fmt=".2f", cmap="YlGnBu",
               xticklabels=[f"Doc {i+1}" for i in range(len(documents))],
               yticklabels=[f"Doc {i+1}" for i in range(len(documents))])
    plt.title("Document Similarity (Cosine Similarity in LSA Space)")
    plt.show()
    
    return X, U, S, Vt, X_lsa, terms, documents

# Demonstrate Latent Semantic Analysis
tdm, U_lsa, S_lsa, Vt_lsa, X_lsa, terms_lsa, docs_lsa = demonstrate_lsa()

# %% [markdown]
# ## 4. Recommendation Systems
# 
# SVD can be used for collaborative filtering in recommendation systems. The idea is to decompose a user-item interaction matrix into user and item feature matrices, which can then be used to predict missing entries (i.e., potential recommendations).
# 
# This approach is often called "matrix factorization" in the context of recommendation systems. Let's implement a simple SVD-based recommendation system:

# %%
def create_user_item_matrix(n_users=10, n_items=8, density=0.6, noise_level=0.1):
    """Create a synthetic user-item rating matrix with some missing values."""
    # Create latent factors (true underlying structure)
    n_factors = 3
    user_factors = np.random.randn(n_users, n_factors)
    item_factors = np.random.randn(n_items, n_factors)
    
    # Generate full matrix
    true_ratings = user_factors @ item_factors.T
    
    # Scale to rating range (1-5)
    true_ratings = 3 + 2 * (true_ratings - np.min(true_ratings)) / (np.max(true_ratings) - np.min(true_ratings))
    
    # Add noise
    ratings = true_ratings + noise_level * np.random.randn(n_users, n_items)
    ratings = np.clip(ratings, 1, 5)
    
    # Set some entries to be missing
    mask = np.random.rand(n_users, n_items) < density
    ratings_with_missing = ratings * mask
    
    # Replace 0s with NaNs to indicate missing values
    ratings_with_missing[~mask] = np.nan
    
    return ratings_with_missing, true_ratings

def svd_for_recommendation(ratings_matrix, k=None):
    """
    Use SVD for collaborative filtering in recommendations.
    
    Args:
        ratings_matrix: User-item matrix with missing values (NaNs)
        k: Number of singular values to use
        
    Returns:
        completed_matrix: Matrix with predictions for missing values
        U, S, Vt: SVD components of the imputed matrix
    """
    # Create a copy of the ratings matrix
    ratings = ratings_matrix.copy()
    
    # Replace NaNs with zeros for SVD
    # (This is a simple approach; more sophisticated methods would use mean imputation or iterative approaches)
    ratings_imputed = np.nan_to_num(ratings, nan=0)
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(ratings_imputed, full_matrices=False)
    
    # Use only the top k singular values if specified
    if k is not None:
        U = U[:, :k]
        S = S[:k]
        Vt = Vt[:k, :]
    
    # Reconstruct the matrix
    completed_matrix = U @ np.diag(S) @ Vt
    
    # Replace known values with the original ratings
    completed_matrix[~np.isnan(ratings_matrix)] = ratings_matrix[~np.isnan(ratings_matrix)]
    
    return completed_matrix, U, S, Vt

def evaluate_recommendations(true_ratings, predicted_ratings, mask):
    """Evaluate the quality of recommendations."""
    # Calculate RMSE on the test set (masked entries)
    test_mask = ~mask
    test_ratings = true_ratings[test_mask]
    test_predictions = predicted_ratings[test_mask]
    
    rmse = np.sqrt(np.mean((test_ratings - test_predictions) ** 2))
    return rmse

def demonstrate_recommendation_system():
    """Demonstrate an SVD-based recommendation system."""
    # Create a synthetic user-item matrix
    n_users, n_items = 10, 8
    ratings_matrix, true_ratings = create_user_item_matrix(n_users, n_items)
    
    # Create user and item names for better visualization
    users = [f"User {i+1}" for i in range(n_users)]
    items = [f"Item {i+1}" for i in range(n_items)]
    
    # Visualize the original matrix with missing values
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    mask = ~np.isnan(ratings_matrix)
    sns.heatmap(ratings_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
               mask=~mask, cbar_kws={'label': 'Rating'}, 
               xticklabels=items, yticklabels=users)
    plt.title("Original Ratings Matrix with Missing Values")
    
    # Apply SVD for collaborative filtering
    k = 3  # Number of latent factors
    completed_matrix, U, S, Vt = svd_for_recommendation(ratings_matrix, k=k)
    
    # Visualize the completed matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(completed_matrix, annot=True, fmt=".1f", cmap="YlGnBu", 
               cbar_kws={'label': 'Rating'}, 
               xticklabels=items, yticklabels=users)
    plt.title(f"Completed Ratings Matrix (k={k})")
    
    plt.tight_layout()
    plt.show()
    
    # Evaluate the recommendations for different values of k
    k_values = range(1, min(n_users, n_items) + 1)
    rmse_values = []
    
    for k in k_values:
        completed_k, _, _, _ = svd_for_recommendation(ratings_matrix, k=k)
        rmse = evaluate_recommendations(true_ratings, completed_k, mask)
        rmse_values.append(rmse)
    
    # Plot RMSE vs. number of factors
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, rmse_values, 'o-')
    plt.title("Recommendation Quality vs. Number of Latent Factors")
    plt.xlabel("Number of Factors (k)")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.show()
    
    # Find the optimal number of factors
    best_k = k_values[np.argmin(rmse_values)]
    print(f"Best number of latent factors: {best_k}")
    print(f"Lowest RMSE: {min(rmse_values):.4f}")
    
    # Visualize user and item embeddings
    # Project users and items into a 2D latent space
    k_viz = min(2, min(n_users, n_items))
    completed_viz, U_viz, S_viz, Vt_viz = svd_for_recommendation(ratings_matrix, k=k_viz)
    
    # User embeddings: U * sqrt(S)
    user_emb = U_viz @ np.diag(np.sqrt(S_viz))
    
    # Item embeddings: Vt.T * sqrt(S)
    item_emb = Vt_viz.T @ np.diag(np.sqrt(S_viz))
    
    # Visualize in 2D
    plt.figure(figsize=(10, 8))
    
    # Plot users
    plt.scatter(user_emb[:, 0], user_emb[:, 1], marker='o', s=100, label='Users')
    for i, user in enumerate(users):
        plt.annotate(user, (user_emb[i, 0], user_emb[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Plot items
    plt.scatter(item_emb[:, 0], item_emb[:, 1], marker='^', s=100, label='Items')
    for i, item in enumerate(items):
        plt.annotate(item, (item_emb[i, 0], item_emb[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Users and Items in Latent Space")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Generate top recommendations for each user
    top_n = 3
    recommendations = {}
    
    for i, user in enumerate(users):
        # Get items the user hasn't rated
        unrated_items = [j for j in range(n_items) if np.isnan(ratings_matrix[i, j])]
        
        if unrated_items:
            # Get predictions for unrated items
            predictions = [(items[j], completed_matrix[i, j]) for j in unrated_items]
            
            # Sort by prediction value
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N recommendations
            top_recommendations = predictions[:top_n]
            recommendations[user] = top_recommendations
    
    # Display recommendations
    print("\nTop Recommendations for Each User:")
    for user, recs in recommendations.items():
        print(f"{user}:")
        for item, score in recs:
            print(f"  {item}: {score:.2f}")
    
    return ratings_matrix, completed_matrix, U, S, Vt, rmse_values

# Demonstrate recommendation system
ratings_orig, ratings_completed, U_rec, S_rec, Vt_rec, rmse_rec = demonstrate_recommendation_system()

# %% [markdown]
# ## 5. Signal Processing and Denoising
# 
# SVD can be used for signal processing tasks such as denoising. By keeping only the most significant singular values, we can filter out noise while preserving the important features of the signal.
# 
# Let's demonstrate SVD-based denoising on a signal:

# %%
def create_noisy_signal(n_samples=1000, noise_level=0.2):
    """Create a synthetic signal with added noise."""
    t = np.linspace(0, 10, n_samples)
    
    # Create a clean signal with multiple frequencies
    signal_clean = (
        np.sin(2 * np.pi * 0.5 * t) +  # 0.5 Hz component
        0.5 * np.sin(2 * np.pi * 1.5 * t) +  # 1.5 Hz component
        0.3 * np.sin(2 * np.pi * 3.0 * t)    # 3.0 Hz component
    )
    
    # Add noise
    noise = noise_level * np.random.randn(n_samples)
    signal_noisy = signal_clean + noise
    
    return t, signal_clean, signal_noisy

def svd_denoise_signal(signal, window_size=20, n_components=5):
    """
    Denoise a signal using SVD.
    
    Args:
        signal: Input signal (1D array)
        window_size: Size of the window for constructing the trajectory matrix
        n_components: Number of singular values/vectors to keep
        
    Returns:
        denoised_signal: Denoised signal
        trajectory_matrix: Trajectory matrix
        U, S, Vt: SVD components of the trajectory matrix
    """
    n_samples = len(signal)
    
    # Construct the trajectory matrix (Hankel matrix)
    n_cols = n_samples - window_size + 1
    trajectory_matrix = np.zeros((window_size, n_cols))
    
    for i in range(n_cols):
        trajectory_matrix[:, i] = signal[i:i+window_size]
    
    # Apply SVD to the trajectory matrix
    U, S, Vt = np.linalg.svd(trajectory_matrix, full_matrices=False)
    
    # Keep only the top n_components
    U_k = U[:, :n_components]
    S_k = S[:n_components]
    Vt_k = Vt[:n_components, :]
    
    # Reconstruct the trajectory matrix
    trajectory_matrix_denoised = U_k @ np.diag(S_k) @ Vt_k
    
    # Convert back to a signal by averaging along anti-diagonals
    denoised_signal = np.zeros(n_samples)
    count = np.zeros(n_samples)
    
    for i in range(window_size):
        for j in range(n_cols):
            denoised_signal[i+j] += trajectory_matrix_denoised[i, j]
            count[i+j] += 1
    
    denoised_signal /= count
    
    return denoised_signal, trajectory_matrix, U, S, Vt

def demonstrate_signal_denoising():
    """Demonstrate signal denoising using SVD."""
    # Create a noisy signal
    t, signal_clean, signal_noisy = create_noisy_signal(n_samples=1000, noise_level=0.2)
    
    # Visualize the clean and noisy signals
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal_clean, 'b-', label='Clean Signal')
    plt.plot(t, signal_noisy, 'r-', alpha=0.5, label='Noisy Signal')
    plt.title("Original and Noisy Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Apply SVD-based denoising
    window_size = 20
    n_components = 3
    signal_denoised, trajectory_matrix, U, S, Vt = svd_denoise_signal(
        signal_noisy, window_size=window_size, n_components=n_components)
    
    # Visualize the denoised signal
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal_clean, 'b-', label='Clean Signal')
    plt.plot(t, signal_noisy, 'r-', alpha=0.3, label='Noisy Signal')
    plt.plot(t, signal_denoised, 'g-', label='Denoised Signal')
    plt.title(f"Signal Denoising with SVD (window_size={window_size}, n_components={n_components})")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate error metrics
    noise_error = np.sqrt(np.mean((signal_clean - signal_noisy) ** 2))
    denoised_error = np.sqrt(np.mean((signal_clean - signal_denoised) ** 2))
    
    print(f"RMS error of noisy signal: {noise_error:.4f}")
    print(f"RMS error of denoised signal: {denoised_error:.4f}")
    print(f"Improvement: {(noise_error - denoised_error) / noise_error:.1%}")
    
    # Visualize trajectory matrix
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(trajectory_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='Amplitude')
    plt.title("Trajectory Matrix (Hankel Matrix)")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    
    # Plot singular values
    plt.subplot(2, 2, 2)
    plt.semilogy(S, 'o-')
    plt.axhline(y=S[n_components-1], color='r', linestyle='--', 
               label=f'Threshold (top {n_components})')
    plt.title("Singular Values (Log Scale)")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot frequency response
    plt.subplot(2, 2, 3)
    freq_clean = np.abs(np.fft.rfft(signal_clean))
    freq_noisy = np.abs(np.fft.rfft(signal_noisy))
    freq_denoised = np.abs(np.fft.rfft(signal_denoised))
    freqs = np.fft.rfftfreq(len(signal_clean), d=(t[1]-t[0]))
    
    plt.plot(freqs, freq_clean, 'b-', label='Clean')
    plt.plot(freqs, freq_noisy, 'r-', alpha=0.5, label='Noisy')
    plt.plot(freqs, freq_denoised, 'g-', label='Denoised')
    plt.title("Frequency Domain Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)  # Focus on relevant frequency range
    
    # Experiment with different numbers of components
    plt.subplot(2, 2, 4)
    n_components_range = range(1, 11)
    errors = []
    
    for n in n_components_range:
        signal_denoised_n, _, _, _, _ = svd_denoise_signal(
            signal_noisy, window_size=window_size, n_components=n)
        error = np.sqrt(np.mean((signal_clean - signal_denoised_n) ** 2))
        errors.append(error)
    
    plt.plot(n_components_range, errors, 'o-')
    plt.title("Error vs. Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("RMS Error")
    plt.grid(True, alpha=0.3)
    plt.xticks(n_components_range)
    
    plt.tight_layout()
    plt.show()
    
    return t, signal_clean, signal_noisy, signal_denoised, S

# Demonstrate signal denoising
t_signal, signal_clean, signal_noisy, signal_denoised, S_signal = demonstrate_signal_denoising()

# %% [markdown]
# ## 6. Pseudoinverse and Least Squares Solutions
# 
# SVD provides a robust way to compute the pseudoinverse of a matrix, which is useful for solving least squares problems and inverting ill-conditioned or non-square matrices.
# 
# The pseudoinverse of a matrix $A$ is given by:
# 
# $$A^+ = V \Sigma^+ U^T$$
# 
# where $\Sigma^+$ is formed by taking the reciprocal of each non-zero singular value in $\Sigma$ and leaving the zeros as zeros.
# 
# Let's demonstrate how SVD can be used to solve linear systems:

# %%
def compute_pseudoinverse(A, tol=1e-10):
    """
    Compute the pseudoinverse of a matrix using SVD.
    
    Args:
        A: Input matrix
        tol: Tolerance for zero singular values
        
    Returns:
        A_pinv: Pseudoinverse of A
    """
    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    
    # Compute reciprocals of singular values, with zeros for small values
    S_inv = np.zeros_like(S)
    S_inv[S > tol] = 1.0 / S[S > tol]
    
    # Compute pseudoinverse
    A_pinv = Vt.T @ np.diag(S_inv) @ U.T
    
    return A_pinv

def svd_least_squares(A, b, tol=1e-10):
    """
    Solve the least squares problem min ||Ax - b||_2 using SVD.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        tol: Tolerance for zero singular values
        
    Returns:
        x: Solution
        residual: Residual norm ||Ax - b||_2
    """
    # Compute pseudoinverse
    A_pinv = compute_pseudoinverse(A, tol)
    
    # Solve the system
    x = A_pinv @ b
    
    # Compute residual
    residual = np.linalg.norm(A @ x - b)
    
    return x, residual

def demonstrate_pseudoinverse():
    """Demonstrate pseudoinverse and least squares solutions using SVD."""
    # Create an overdetermined system (more equations than unknowns)
    m, n = 10, 3  # 10 equations, 3 unknowns
    np.random.seed(42)
    
    # Create a coefficient matrix
    A = np.random.randn(m, n)
    
    # Create a true solution
    x_true = np.random.randn(n)
    
    # Create the right-hand side (with some noise)
    noise_level = 0.1
    b_clean = A @ x_true
    b = b_clean + noise_level * np.random.randn(m)
    
    # Solve using SVD pseudoinverse
    x_svd, residual_svd = svd_least_squares(A, b)
    
    # For comparison, solve using NumPy's least squares function
    x_np, residual_np, rank_np, s_np = np.linalg.lstsq(A, b, rcond=None)
    
    # Print the results
    print("Least Squares Solutions:")
    print("-" * 50)
    print(f"{'Method':<15} {'Solution':<30} {'Residual':<15}")
    print("-" * 50)
    print(f"{'True':<15} {np.array2string(x_true, precision=4):<30} {'N/A':<15}")
    print(f"{'SVD':<15} {np.array2string(x_svd, precision=4):<30} {residual_svd:.6f}")
    print(f"{'NumPy':<15} {np.array2string(x_np, precision=4):<30} {residual_np:.6f}")
    
    # Visualize the solutions
    plt.figure(figsize=(10, 6))
    width = 0.3
    indices = np.arange(n)
    
    plt.bar(indices - width, x_true, width, label='True')
    plt.bar(indices, x_svd, width, label='SVD')
    plt.bar(indices + width, x_np, width, label='NumPy')
    
    plt.title("Least Squares Solutions")
    plt.xlabel("Variable Index")
    plt.ylabel("Value")
    plt.xticks(indices)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Visualize the residuals
    plt.figure(figsize=(10, 6))
    
    residual_vector_svd = A @ x_svd - b
    residual_vector_np = A @ x_np - b
    
    plt.plot(residual_vector_svd, 'o-', label='SVD')
    plt.plot(residual_vector_np, 's-', label='NumPy')
    
    plt.title("Residuals")
    plt.xlabel("Equation Index")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Now demonstrate with an ill-conditioned matrix
    A_ill = np.array([
        [1, 1, 1],
        [1, 1, 1.001],
        [1, 1.001, 1],
        [1.001, 1, 1]
    ])
    
    # Create a right-hand side
    b_ill = np.array([1, 2, 3, 4])
    
    # Compute the condition number
    _, S_ill, _ = np.linalg.svd(A_ill)
    cond_num = S_ill[0] / S_ill[-1]
    
    print(f"\nIll-conditioned matrix with condition number: {cond_num:.1e}")
    
    # Solve using different tolerances
    tolerances = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    solutions = []
    residuals = []
    
    for tol in tolerances:
        x_tol, residual_tol = svd_least_squares(A_ill, b_ill, tol=tol)
        solutions.append(x_tol)
        residuals.append(residual_tol)
    
    # Visualize the effect of the tolerance
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    for i, (tol, x) in enumerate(zip(tolerances, solutions)):
        plt.plot(x, 'o-', label=f'Tolerance = {tol:.1e}')
    
    plt.title("Solutions for Different Tolerances")
    plt.xlabel("Variable Index")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogx(tolerances, residuals, 'o-')
    plt.title("Residual vs. Tolerance")
    plt.xlabel("Tolerance")
    plt.ylabel("Residual Norm")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the singular values of the ill-conditioned matrix
    plt.figure(figsize=(10, 6))
    plt.semilogy(S_ill, 'o-')
    
    for tol in tolerances:
        plt.axhline(y=tol, linestyle='--', alpha=0.5, label=f'Tolerance = {tol:.1e}')
    
    plt.title("Singular Values of Ill-conditioned Matrix")
    plt.xlabel("Index")
    plt.ylabel("Value (log scale)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    return A, b, x_true, x_svd, A_ill, b_ill, solutions, residuals

# Demonstrate pseudoinverse and least squares
A_ls, b_ls, x_true_ls, x_svd_ls, A_ill, b_ill, solutions_ill, residuals_ill = demonstrate_pseudoinverse()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've explored a wide range of applications of Singular Value Decomposition (SVD) across different domains:
# 
# 1. **Image Compression**: We showed how SVD enables efficient image compression by keeping only the most significant singular values and their corresponding vectors.
# 
# 2. **Principal Component Analysis (PCA)**: We demonstrated how SVD forms the mathematical foundation for PCA, a widely-used dimensionality reduction technique.
# 
# 3. **Latent Semantic Analysis (LSA)**: We explored how SVD can uncover hidden semantic relationships in text data, enabling document similarity analysis and topic modeling.
# 
# 4. **Recommendation Systems**: We implemented a collaborative filtering algorithm using SVD to predict user preferences and generate recommendations.
# 
# 5. **Signal Processing and Denoising**: We applied SVD to denoise signals by separating signal components from noise.
# 
# 6. **Pseudoinverse and Least Squares**: We showed how SVD provides a robust way to compute the pseudoinverse of a matrix, which is useful for solving linear systems and least squares problems.
# 
# These applications highlight the versatility and power of SVD as a fundamental tool in data analysis, machine learning, signal processing, and many other fields. The ability to decompose a matrix into its principal components makes SVD a cornerstone technique in computational linear algebra.

# %%