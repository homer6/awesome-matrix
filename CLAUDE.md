# CLAUDE.md - Assistant Guide

## Project: awesome-matrix
An educational collection of matrix operations and utilities with PyTorch examples to develop intuition around tensors and n-dimensional thinking.

## Commands
- **Run examples:** `python src/examples/[example_name].py`
- **Run tests:** `pytest tests/`
- **Run single test:** `pytest tests/test_file.py::test_function`
- **Install deps:** `pip install -r requirements.txt`
- **Compile examples:** `make compile_example EXAMPLE=example_name`
- **Execute notebooks:** `make execute_example EXAMPLE=example_name`
- **Compile and execute all:** `make compile_and_execute`
- **Clean notebooks:** `make clean`

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