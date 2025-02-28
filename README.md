# Awesome Matrix Operations

A curated list of awesome matrix operations, tutorials, visualizations, and resources focused on building intuition for n-dimensional thinking. This repository emphasizes educational content with PyTorch examples.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Getting Started

For installation instructions and setup guidance, see the [Installation Guide](INSTALL.md).

To explore example code and demonstrations, check out the [examples directory](examples/).

Source code for all examples is maintained in the [src/examples](src/examples/) directory. See [Compiling Examples](docs/COMPILE.md) for information on how examples are compiled from source to notebooks.

## Contents

- [Basics](#basics)
- [Linear Algebra Fundamentals](#linear-algebra-fundamentals)
- [Matrix Decompositions](#matrix-decompositions)
- [Tensor Operations](#tensor-operations)
- [Visualization Techniques](#visualization-techniques)
- [Applications](#applications)
- [Numerical Stability and Precision](#numerical-stability-and-precision)
- [Performance and Optimization](#performance-and-optimization)
- [Matrix Calculus and Automatic Differentiation](#matrix-calculus-and-automatic-differentiation)
- [Randomized Algorithms](#randomized-algorithms)
- [Matrix-based Statistical Methods](#matrix-based-statistical-methods)
- [Sparse and Structured Matrices](#sparse-and-structured-matrices)
- [High-Performance Computing and Specialized Libraries](#high-performance-computing-and-specialized-libraries)
- [Learning Resources](#learning-resources)
- [Libraries and Tools](#libraries-and-tools)
- [Research Papers](#research-papers)
- [Contributing](#contributing)
- [License](#license)

## Basics

*Fundamental concepts for understanding matrices*

- Matrix creation and indexing  
  - Creating matrices with PyTorch: zeros, ones, random, eye, diag  
  - Slicing and indexing techniques  
  - Boolean indexing and masking  
- Shapes and dimensions  
  - Understanding tensor shapes  
  - Broadcasting semantics  
  - Shape manipulation (view, reshape, squeeze, unsqueeze)  
- Basic operations  
  - Element-wise operations  
  - Matrix transposition  
  - Concatenation and stacking  
- Memory layout and optimization  
  - Row-major vs column-major storage  
  - Stride and contiguity  
  - In-place operations  
- Data types and precision  
  - Floating-point vs integer matrices  
  - Precision considerations (float32, float64)  
  - Mixed precision operations  

## Linear Algebra Fundamentals

*Essential operations for matrix manipulation*

- [Matrix multiplication](examples/01_matrix_multiplication/)  
  - Standard matrix product (matmul)  
  - Batch matrix multiplication  
  - Optimizations and computational complexity  
- [Dot and inner products](examples/02_dot_and_inner_products/)  
  - Vector dot products  
  - Matrix-vector products  
  - Different inner product definitions  
- [Cross products and outer products](examples/03_cross_and_outer_products/)  
  - Geometric interpretation  
  - Applications in physics  
  - Kronecker product  
- [Eigenvalues and eigenvectors](examples/04_eigenvalues_eigenvectors/)  
  - Intuitive explanation and visualization  
  - Power iteration method  
  - Applications in PCA and spectral methods  
- [Vector spaces and subspaces](examples/05_vector_spaces_subspaces/)  
  - Basis and dimensionality  
  - Orthogonality and projection  
  - Change of basis  
- [Matrix properties](examples/06_matrix_properties/)  
  - Determinant and trace  
  - Rank and nullity  
  - Positive definiteness  
  - Condition number  
- Special matrices  
  - Symmetric, skew-symmetric  
  - Orthogonal and unitary  
  - Toeplitz and circulant  
  - Sparse matrices  

## Matrix Decompositions

*Techniques to factorize matrices*

- [LU decomposition](examples/07_lu_decomposition/)  
  - Gaussian elimination  
  - Computational aspects  
  - Solving linear systems  
- [QR decomposition](examples/08_qr_decomposition/)  
  - Gram-Schmidt process  
  - Householder reflections  
  - Applications in least squares  
- [Singular Value Decomposition (SVD)](examples/09_singular_value_decomposition/)  
  - Geometric interpretation  
  - Low-rank approximation  
  - Image compression with SVD  
  - Truncated SVD  
- Eigendecomposition  
  - Relationship with SVD  
  - Symmetric eigendecomposition  
  - Power method and inverse iteration  
- [Cholesky decomposition](examples/10_cholesky_decomposition/)  
  - Properties of positive definite matrices  
  - Applications in sampling multivariate distributions  
  - Relationship to LU decomposition  
- Schur decomposition  
  - Triangularization  
  - Applications in stability analysis  
- Polar decomposition  
  - Rotation and scaling components  
  - Applications in computer graphics  

## Tensor Operations

*Working with higher-dimensional data*

- [Tensor contraction and Einstein notation](examples/11_tensor_operations_einsum/)  
  - Summation over indices  
  - Relationship with matrix multiplication  
  - Example: CNN as tensor contraction  
  - Simplifying complex operations  
  - Index notation for tensors  
  - PyTorch einsum examples  
- Reshaping and permuting dimensions  
  - Transpose generalizations  
  - View vs reshape  
  - Permutation operations  
- Batch operations  
  - Vectorized computation  
  - Broadcasting for batch processing  
  - Parallel computation considerations  
- Advanced PyTorch tensor operations  
  - Unfold and fold  
  - Sparse tensors  
  - Autograd with tensors  
- Tensor networks  
  - Matrix product states  
  - Tensor train decomposition  
  - Applications in quantum physics and machine learning  

## Visualization Techniques

*Methods to visualize matrices and operations*

- [Visualization Techniques](examples/12_visualization_techniques/)  
  - Heatmaps and colormaps  
  - Visualizing matrix values  
  - Correlation matrices  
  - Attention weights visualization  
  - 3D visualizations of transformations  
  - Linear transformations as geometric operations  
  - Visualizing eigenspaces  
  - Interactive transformation demos  
- Animation of matrix operations  
  - Iterative algorithms  
  - Dynamical systems  
  - SVD stepwise visualization  
- Dimensional reduction for visualization  
  - PCA and t-SNE for high-dimensional data  
  - Vector field visualization  
  - Embeddings visualization  
- Network and graph representations  
  - Adjacency matrices  
  - Graph Laplacians  
  - Spectral clustering visualization  

## Applications

*Real-world uses of matrix operations*

- Computer graphics and 3D transformations  
  - Rotation, translation, and scaling matrices  
  - Projection matrices  
  - Quaternions for rotation  
  - Homogeneous coordinates  
- Machine learning and deep neural networks  
  - Weight matrices and biases  
  - Attention mechanisms  
  - Convolutional filters as matrices  
  - Natural language processing embeddings  
  - Graph Neural Networks (GNNs) for structured data
- Signal processing  
  - Fourier transforms as matrix operations  
  - Filter banks  
  - Wavelets transform matrices  
  - Image processing kernels  
- Quantum computing  
  - Quantum gates as matrices  
  - Density matrices  
  - Entanglement measurement  
- Physical simulations  
  - Rigid body dynamics  
  - Fluid dynamics discretization  
  - Finite element methods  
  - PDE-based matrix approaches (e.g., building large sparse system matrices)
- Optimization  
  - Gradient descent with matrices  
  - Hessian and Fisher information  
  - Second-order methods  
- Recommender systems  
  - Matrix factorization techniques  
  - Collaborative filtering as matrix completion  
  - Implicit vs explicit feedback matrices  

## Numerical Stability and Precision

*Handling floating-point concerns*

- Precision loss and numerical errors  
- Conditioning and stability analysis  
- Strategies to handle ill-conditioned matrices  

## Performance and Optimization

*Making matrix operations faster*

- Efficient computation on CPUs and GPUs  
  - Vectorization and parallelization  
  - Multi-GPU strategies for large-scale matrix problems  
- Profiling and benchmarking matrix operations  
  - Using PyTorch’s profiler and other tools  
  - Memory usage optimization  
- Optimization strategies for large-scale problems  
  - Block matrix algorithms  
  - Out-of-core methods  
- High-Performance Data Structures  
  - Effective memory alignment (strides, tiling)  
  - Pointer manipulations and HPC data layouts  

## Matrix Calculus and Automatic Differentiation

*Gradient-based methods in matrix form*

- Matrix calculus fundamentals  
- Jacobians and Hessians with PyTorch autograd  
- Practical backpropagation examples  

## Randomized Algorithms

*Probabilistic approaches to matrix problems*

- Randomized matrix decompositions  
- Approximation algorithms and matrix sketching  
- Randomized PCA and SVD  

## Matrix-based Statistical Methods

*Multivariate data analysis*

- Covariance and correlation matrices  
- Multivariate statistical methods using matrix algebra  
- Principal Component Analysis (PCA) and beyond  

## Sparse and Structured Matrices

*Handling specialized matrix formats*

- Sparse matrix representations and optimizations  
- Structured matrices in graph algorithms and network analysis  
- Graph-based operations for GNNs  
- Block diagonal, banded, Toeplitz, and other special structures  

---

## High-Performance Computing and Specialized Libraries

*Getting the most out of large-scale matrix operations*

- Specialized HPC Libraries  
  - Intel MKL, OpenBLAS, cuBLAS  
  - MAGMA, Kokkos, and other frameworks  
  - Integrating these libraries with PyTorch or NumPy  
- Extensions of PyTorch  
  - PyTorch/XLA for TPUs  
  - PyTorch extensions for real-time GPU notebooks  
  - Third-party HPC-oriented PyTorch add-ons  
- Distributed Computing  
  - MPI-based solutions for distributed matrices  
  - Dask, Spark, and other parallel frameworks  
- Real-Time and Streaming Applications  
  - Online matrix computations  
  - GPU-based dashboards and interactive notebooks  

---

## Learning Resources

*Tutorials, books, and courses*

- Interactive tutorials  
  - Matrix visualization platforms  
  - Jupyter notebook collections  
  - Interactive eigenvalue demos  
- Recommended books  
  - Linear algebra textbooks  
  - Numerical computing references  
  - Machine learning matrix mathematics  
- Online courses  
  - Linear algebra foundations  
  - Computational linear algebra  
  - Deep learning matrix mathematics  
- Visualization tools  
  - Matrix explorer applications  
  - Tensor visualization libraries  
  - Interactive transformation tools  
- YouTube channels and video series  
  - 3Blue1Brown linear algebra series  
  - MIT OpenCourseWare  
  - PyTorch tutorials  

## Libraries and Tools

*Software for matrix operations*

- PyTorch  
  - Core tensor operations  
  - GPU acceleration  
  - Automatic differentiation  
- NumPy and SciPy  
  - Fundamental matrix operations  
  - Scientific computing routines  
  - Specialized matrix classes  
- JAX  
  - Accelerated linear algebra  
  - Functional transformations  
  - AutoDiff capabilities  
- MATLAB and alternatives  
  - Commercial matrix libraries  
  - Open source alternatives  
  - Domain-specific toolboxes  
- Specialized libraries  
  - Sparse matrix libraries  
  - Big data matrix computations  
  - Quantum tensor networks  
  - HPC-oriented libraries (e.g., Kokkos, Trilinos, PETSc)

## Research Papers

*Academic publications on matrix methods*

- Foundational papers  
  - Matrix decomposition algorithms  
  - Numerical stability  
  - Complexity analysis  
- Recent advances  
  - Randomized algorithms  
  - Tensor methods  
  - Quantum-inspired algorithms  
- Review papers and tutorials  
  - State-of-the-art surveys  
  - Educational publications  
  - Comparative analysis  

## Contributing

Contributions are welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

## License

See [LICENSE](LICENSE) for details.
