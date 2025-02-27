.PHONY: all compile_notebooks compile_and_execute clean

EXAMPLE_SOURCES := $(wildcard src/examples/*.py)
EXAMPLE_NAMES := $(basename $(notdir $(EXAMPLE_SOURCES)))
EXAMPLE_NOTEBOOKS := $(addprefix examples/, $(addsuffix .ipynb, $(EXAMPLE_NAMES)))

# Main target compiles and executes all notebooks
all: compile_and_execute

# Compile all notebooks without executing them
compile_notebooks: $(EXAMPLE_NOTEBOOKS)

# Compile a specific notebook by example name (e.g., make compile_example EXAMPLE=matrix_multiplication)
compile_example:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make compile_example EXAMPLE=example_name"; \
		exit 1; \
	fi
	@./bin/compile_examples.py $(EXAMPLE)
	@echo "Compiled examples/$(EXAMPLE).ipynb"

# Compile and execute all notebooks
compile_and_execute: compile_notebooks
	@echo "Executing all notebooks..."
	@for notebook in $(EXAMPLE_NOTEBOOKS); do \
		echo "Executing $$notebook..."; \
		jupyter nbconvert --execute --to notebook --inplace $$notebook; \
	done
	@echo "All notebooks compiled and executed successfully!"

# Execute a specific notebook (e.g., make execute_example EXAMPLE=matrix_multiplication)
execute_example:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make execute_example EXAMPLE=example_name"; \
		exit 1; \
	fi
	@jupyter nbconvert --execute --to notebook --inplace examples/$(EXAMPLE).ipynb
	@echo "Executed examples/$(EXAMPLE).ipynb"

# Rule to compile an individual notebook from its source
examples/%.ipynb: src/examples/%.py
	@echo "Compiling $< to $@..."
	@./bin/compile_examples.py $(basename $(notdir $<))

# Clean all compiled notebooks
clean:
	@echo "Removing all compiled notebooks..."
	@rm -f $(EXAMPLE_NOTEBOOKS)
	@echo "Notebooks removed!"

help:
	@echo "Available targets:"
	@echo "  all                  - Compile and execute all notebooks (default)"
	@echo "  compile_notebooks    - Compile all notebooks without executing them"
	@echo "  compile_example      - Compile a specific notebook (e.g., make compile_example EXAMPLE=matrix_multiplication)"
	@echo "  execute_example      - Execute a specific notebook (e.g., make execute_example EXAMPLE=matrix_multiplication)"
	@echo "  compile_and_execute  - Compile and execute all notebooks"
	@echo "  clean                - Remove all compiled notebooks"
	@echo "  help                 - Show this help message"