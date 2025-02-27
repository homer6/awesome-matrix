# Compiling Examples

This document explains how to compile the source examples in `src/examples/` into Jupyter notebooks for easier viewing and interaction.

## Overview

The source code for all examples is maintained in Python files in the `src/examples/` directory. These Python files contain special markers that indicate cell boundaries and cell types, which are then converted into Jupyter notebooks in the `examples/` directory.

## Compilation Process

To compile a Python source file into a Jupyter notebook, use the `compile_examples.py` script:

```bash
# Compile a specific example
./bin/compile_examples.py matrix_multiplication

# Compile all examples
for file in src/examples/*.py; do
  basename=$(basename "$file" .py)
  ./bin/compile_examples.py "$basename"
done
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

## Editing Process

1. **Always edit the source files** in `src/examples/`, never edit the compiled notebooks directly
2. After making changes, recompile to update the notebooks
3. The compiled notebooks include a warning header to remind users not to edit them directly

## Best Practices

1. Start each source file with a markdown cell that explains the purpose of the example
2. Include plenty of comments and explanations
3. Break up code into logical sections with markdown explanations in between
4. Include visualizations and examples that build intuition
5. Follow the format guidelines in CLAUDE.md