# Compiling Examples

This document explains how to compile the source examples in `src/examples/` into Jupyter notebooks for easier viewing and interaction.

## Overview

The source code for all examples is maintained in Python files in the `src/examples/` directory. These Python files contain special markers that indicate cell boundaries and cell types, which are then converted into Jupyter notebooks in the `examples/` directory.

## Compilation Process

A Makefile is provided to simplify the notebook compilation and execution process:

```bash
# Compile all notebooks without executing them
make compile_notebooks

# Execute all notebooks after compiling them
make compile_and_execute

# Compile a specific example
make compile_example EXAMPLE=matrix_multiplication

# Execute a specific example
make execute_example EXAMPLE=matrix_multiplication

# Clean all compiled notebooks
make clean

# Show help
make help
```

You can also manually compile notebooks using the underlying scripts:

```bash
# Compile a specific example
./bin/compile_examples.py matrix_multiplication

# Execute the notebook cells to generate visualizations
jupyter nbconvert --execute --to notebook --inplace examples/matrix_multiplication.ipynb
```

## Source File Format

Source files should use the following format for defining notebook cells:

```python
# %% [markdown]
# # This is a markdown cell
# With multi-line content

# %%
# This is a code cell
import torch
print("Hello, world!")
```

### Cell Type Markers

- `# %%` - Start of a code cell
- `# %% [markdown]` - Start of a markdown cell

For markdown cells, each line should start with `# ` which will be stripped during compilation.

## Notebook Structure Best Practices

1. Start with a markdown cell that introduces the topic
2. Add section headers as markdown cells (using `## Section Title` format)
3. Split code into logical sections with meaningful cell boundaries
4. Include explanatory markdown cells between code sections
5. End with a conclusion or summary

Example structure:
```python
# %% [markdown]
# # Matrix Operations
# Introduction to the topic...

# %% [markdown]
# ## Setup and Imports
# First, we'll import necessary libraries.

# %%
import torch
import numpy as np

# %% [markdown]
# ## Core Function Implementation
# Now we'll implement the main functionality.

# %%
def my_function():
    pass
```

## Editing Process

1. **Always edit the source files** in `src/examples/`, never edit the compiled notebooks directly
2. After making changes, recompile to update the notebooks
3. Execute the notebooks to generate visualizations
4. The compiled notebooks include a warning header to remind users not to edit them directly

## Handling Special Cases

### Main Function 

If your source file has a `main()` function with an `if __name__ == "__main__":` guard at the end, the compiler will automatically:
1. Extract this code from any markdown cell it might be part of
2. Create a separate code cell for this block

### Troubleshooting

If your notebook doesn't compile correctly:
1. Check that cell markers (`# %%` and `# %% [markdown]`) are placed at the start of lines
2. Ensure markdown content is properly prefixed with `# `
3. Verify there are no syntax errors in the Python code

## Best Practices

1. Start each source file with a markdown cell that explains the purpose of the example
2. Include plenty of comments and explanations
3. Break up code into logical sections with markdown explanations in between
4. Include visualizations and examples that build intuition
5. Follow the format guidelines in CLAUDE.md