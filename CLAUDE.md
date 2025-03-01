# CLAUDE.md - Assistant Guide

## Project: awesome-matrix
An educational collection of matrix operations and utilities with PyTorch examples to develop intuition around tensors and n-dimensional thinking.

## Commands
- **Run examples:** `python src/examples/[dir]/[file].py`  
- **Run tests:** `pytest tests/`
- **Run single test:** `pytest tests/test_file.py::test_function`
- **Install deps:** `pip install -r requirements.txt`
- **Compile all examples:** `make compile`
- **Compile example directory:** `make compile_example EXAMPLE=01_matrix_multiplication`
- **Compile specific file:** `make compile_file EXAMPLE=01_matrix_multiplication/01_introduction`
- **Execute all:** `make execute`
- **Execute example directory:** `make execute_example EXAMPLE=01_matrix_multiplication`
- **Execute specific file:** `make execute_file EXAMPLE=01_matrix_multiplication/01_introduction`
- **Compile and execute:** `make compile_and_execute EXAMPLE=01_matrix_multiplication`
- **Clean notebooks:** `make clean`
- **Show help:** `make help`

## Code Style Guidelines
- **Framework:** Use PyTorch for tensor/matrix operations
- **Formatting:** Black formatter with 88 character line length
- **Imports:** Group by standard library, third-party (PyTorch first), then local
- **Types:** Use type hints (PEP 484)
- **Comments:** Include mathematical notation and intuition in docstrings
- **Examples:** Each operation should include visual/graphical explanations
- **Educational:** Focus on building intuition about n-dimensional operations

## Content Organization
- Structure like other awesome-* repos with categorized links
- Each example should include code, visualization, and explanation
- Include real-world applications for each concept
- Emphasize dimensional thinking and tensor shape transformations

## Planned Examples
When building new examples, always refer to the README.md for guidance on what to implement next. The README.md contains a detailed outline of planned examples organized in these categories:

- **Basics**: Matrix creation, shapes, basic operations, memory layout, data types
- **Linear Algebra Fundamentals**: Matrix multiplication, dot products, cross products, eigenvalues, etc.
- **Matrix Decompositions**: LU, QR, SVD, Cholesky, etc.
- **Tensor Operations**: Tensor contraction, Einstein notation, reshaping
- **Visualization Techniques**: Heatmaps, 3D visualizations, animations
- **Applications**: Graphics, ML, signal processing, quantum computing, etc.

Each category in the README.md contains detailed sub-topics that should be implemented as examples. For each topic, reference the README.md to ensure the example covers all the relevant aspects mentioned there.

### Currently Implemented Examples
1. Matrix Multiplication ([01_matrix_multiplication](examples/01_matrix_multiplication/))
2. Dot and Inner Products ([02_dot_and_inner_products](examples/02_dot_and_inner_products/))
3. Cross Products and Outer Products ([03_cross_and_outer_products](examples/03_cross_and_outer_products/))
4. Eigenvalues and Eigenvectors ([04_eigenvalues_eigenvectors](examples/04_eigenvalues_eigenvectors/))
5. Vector Spaces and Subspaces ([05_vector_spaces_subspaces](examples/05_vector_spaces_subspaces/))
6. Matrix Properties ([06_matrix_properties](examples/06_matrix_properties/))
7. LU Decomposition ([07_lu_decomposition](examples/07_lu_decomposition/))
8. QR Decomposition ([08_qr_decomposition](examples/08_qr_decomposition/))
9. Singular Value Decomposition (SVD) ([09_singular_value_decomposition](examples/09_singular_value_decomposition/))
10. Cholesky Decomposition ([10_cholesky_decomposition](examples/10_cholesky_decomposition/))
11. Tensor Operations and Einstein Notation ([11_tensor_operations_einsum](examples/11_tensor_operations_einsum/))
12. Visualization Techniques ([12_visualization_techniques](examples/12_visualization_techniques/))
13. Matrix Basics ([13_matrix_basics](examples/13_matrix_basics/))

### Next Examples to Implement
Based on the README.md and our current progress, the next topics to implement would be:

1. **Applications (14_computer_graphics):** Computer Graphics and 3D Transformations
   - Rotation, translation, and scaling matrices
   - Projection matrices
   - Homogeneous coordinates
   - 3D transformations and applications

2. **Applications (15_machine_learning):** Machine Learning Applications
   - Weight matrices and biases
   - Attention mechanisms
   - Convolutional filters as matrices
   - Neural network transformation visualizations

3. **Advanced Decompositions (16_eigendecomposition):** Eigendecomposition
   - Relationship with SVD
   - Symmetric eigendecomposition
   - Power method and inverse iteration
   - Applications in data analysis

## Example Source Format
- Source code for examples is kept in `src/examples/` directory
- Compiled notebooks are generated in `examples/` directory
- Source files use cell markers for Jupyter notebook compilation:
  - `# %%` - Start of a code cell
  - `# %% [markdown]` - Start of a markdown cell
  - For markdown cells, each content line should start with `# `
- Always edit source files in `src/examples/`, never edit compiled notebooks
- After editing source files, recompile to update notebooks

## Notebook Structure
- Notebooks should start with a markdown cell explaining the concept
- Each major section should have a markdown header (`## Section Title`)
- Code cells should be followed by outputs (visualizations, results)
- Include clear explanations between code sections
- End with a conclusion or summary

## Visualization Guidelines
- All matrix visualizations should include annotations showing values
- Colormap should range from light blue to dark blue for easy readability
- Include row and column indices and dimension information
- Show the full computation process, not just end results
- For matrix multiplication, show input matrices and result side by side

## Development Workflow
1. Edit source files in `src/examples/` with proper cell markers
2. Compile to notebook with `make compile_example EXAMPLE=name`
3. Execute cells with `make execute_example EXAMPLE=name`
4. Verify outputs appear directly below respective code cells
5. Check for proper formatting and ensure educational clarity
6. Do not commit/push directly to main branch unless necessary

## Important Notes
- Never run `git add` automatically; let the user manage git operations
- Focus on developing intuition - prioritize visual explanations over formal math notation
- Show step-by-step processes rather than just end results
- Keep example data small enough to be easily understood visually