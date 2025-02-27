# CLAUDE.md - Assistant Guide

## Project: awesome-matrix
An educational collection of matrix operations and utilities with PyTorch examples to develop intuition around tensors and n-dimensional thinking.

## Commands
- **Run examples:** `python src/examples/[example_name].py`
- **Run tests:** `pytest tests/`
- **Run single test:** `pytest tests/test_file.py::test_function`
- **Install deps:** `pip install -r requirements.txt`
- **Compile examples:** `./bin/compile_examples.py [example_name]`
- **Compile all examples:** `for file in src/examples/*.py; do basename=$(basename "$file" .py); ./bin/compile_examples.py "$basename"; done`

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

## Example Source Format
- Source code for examples is kept in `src/examples/` directory
- Compiled notebooks are generated in `examples/` directory
- Source files use cell markers for Jupyter notebook compilation:
  - `# %%` - Start of a code cell
  - `# %% [markdown]` - Start of a markdown cell
  - For markdown cells, each content line should start with `# `
- Always edit source files in `src/examples/`, never edit compiled notebooks
- After editing source files, recompile to update notebooks