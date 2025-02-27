# CLAUDE.md - Assistant Guide

## Project: awesome-matrix
An educational collection of matrix operations and utilities with PyTorch examples to develop intuition around tensors and n-dimensional thinking.

## Commands
- **Run examples:** `python examples/[example_name].py`
- **Run tests:** `pytest tests/`
- **Run single test:** `pytest tests/test_file.py::test_function`
- **Install deps:** `pip install -r requirements.txt`

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