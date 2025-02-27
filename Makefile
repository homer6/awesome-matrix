.PHONY: all compile compile_example execute clean help

# Main target compiles and executes all examples
all: compile execute

# Compile all examples
compile:
	@echo "Compiling all examples..."
	@for dir in $$(find src/examples -mindepth 1 -maxdepth 1 -type d | sort); do \
		example_name=$$(basename $$dir); \
		echo "Compiling $$example_name..."; \
		./bin/compile_examples.py $$example_name; \
	done
	@echo "All examples compiled successfully!"

# Compile a specific example (e.g., make compile_example EXAMPLE=01_matrix_multiplication)
compile_example:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make compile_example EXAMPLE=example_name"; \
		echo "Examples: 01_matrix_multiplication, 02_matrix_decomposition, etc."; \
		exit 1; \
	fi
	@./bin/compile_examples.py $(EXAMPLE)
	@echo "Compiled $(EXAMPLE) examples"

# Compile a specific file (e.g., make compile_file EXAMPLE=01_matrix_multiplication/01_introduction)
compile_file:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make compile_file EXAMPLE=directory/filename"; \
		echo "Example: make compile_file EXAMPLE=01_matrix_multiplication/01_introduction"; \
		exit 1; \
	fi
	@./bin/compile_examples.py $(EXAMPLE)
	@echo "Compiled $(EXAMPLE).ipynb"

# Execute all compiled notebooks
execute:
	@echo "Executing all notebooks..."
	@find examples -name "*.ipynb" -type f | sort | xargs -I{} sh -c 'echo "Executing {}..." && jupyter nbconvert --execute --to notebook --inplace {}'
	@echo "All notebooks executed successfully!"

# Execute a specific notebook (e.g., make execute_example EXAMPLE=01_matrix_multiplication)
execute_example:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make execute_example EXAMPLE=example_name"; \
		echo "Examples: 01_matrix_multiplication, 02_matrix_decomposition, etc."; \
		exit 1; \
	fi
	@echo "Executing examples/$(EXAMPLE)/*.ipynb..."
	@find examples/$(EXAMPLE) -name "*.ipynb" -type f | sort | xargs -I{} sh -c 'echo "Executing {}..." && jupyter nbconvert --execute --to notebook --inplace {}'
	@echo "Executed $(EXAMPLE) notebooks"

# Execute a specific file (e.g., make execute_file EXAMPLE=01_matrix_multiplication/01_introduction)
execute_file:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make execute_file EXAMPLE=directory/filename"; \
		echo "Example: make execute_file EXAMPLE=01_matrix_multiplication/01_introduction"; \
		exit 1; \
	fi
	@echo "Executing examples/$(EXAMPLE).ipynb..."
	@jupyter nbconvert --execute --to notebook --inplace examples/$(EXAMPLE).ipynb
	@echo "Executed examples/$(EXAMPLE).ipynb"

# Compile and execute in one step
compile_and_execute:
	@if [ -z "$(EXAMPLE)" ]; then \
		echo "Usage: make compile_and_execute EXAMPLE=example_name"; \
		exit 1; \
	fi
	@./bin/compile_examples.py $(EXAMPLE) --execute
	@echo "Compiled and executed $(EXAMPLE)"

# Clean all compiled notebooks
clean:
	@echo "Removing all compiled notebooks..."
	@find examples -name "*.ipynb" -type f -delete
	@echo "Notebooks removed!"

help:
	@echo "Available targets:"
	@echo "  all                  - Compile and execute all notebooks (default)"
	@echo "  compile              - Compile all notebooks without executing them"
	@echo "  execute              - Execute all compiled notebooks"
	@echo "  compile_example      - Compile examples in a directory (e.g., make compile_example EXAMPLE=01_matrix_multiplication)"
	@echo "  compile_file         - Compile a specific file (e.g., make compile_file EXAMPLE=01_matrix_multiplication/01_introduction)"
	@echo "  execute_example      - Execute notebooks in a directory (e.g., make execute_example EXAMPLE=01_matrix_multiplication)"
	@echo "  execute_file         - Execute a specific notebook (e.g., make execute_file EXAMPLE=01_matrix_multiplication/01_introduction)"
	@echo "  compile_and_execute  - Compile and execute a specific example (e.g., make compile_and_execute EXAMPLE=01_matrix_multiplication)"
	@echo "  clean                - Remove all compiled notebooks"
	@echo "  help                 - Show this help message"