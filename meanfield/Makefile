# Makefile for Mean-Field QLDPC Decoder

.PHONY: all build clean test install uninstall help

# Default target
all: build

# Build the extension
build:
	@echo "Building mean-field decoder extension..."
	python setup.py build_ext --inplace
	@echo "Build completed successfully!"

# Install the package
install:
	@echo "Installing mean-field decoder..."
	pip install -e .
	@echo "Installation completed!"

# Install in development mode
install-dev:
	@echo "Installing mean-field decoder in development mode..."
	pip install -e .[dev]
	@echo "Development installation completed!"

# Uninstall the package
uninstall:
	@echo "Uninstalling mean-field decoder..."
	pip uninstall -y meanfield_decoder
	@echo "Uninstallation completed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f *.so
	rm -f *.pyd
	rm -f meanfield_analysis.png
	@echo "Clean completed!"

# Run tests
test:
	@echo "Running tests..."
	python -m pytest tests/ -v

# Run example
example:
	@echo "Running example..."
	python example.py

# Run benchmark
benchmark:
	@echo "Running benchmarks..."
	python -m pytest tests/ --benchmark-only

# Check code style
lint:
	@echo "Checking code style..."
	flake8 *.py
	pylint *.py

# Show help
help:
	@echo "Available targets:"
	@echo "  build       - Build the C++ extension"
	@echo "  install     - Install the package"
	@echo "  install-dev - Install with development dependencies"
	@echo "  uninstall   - Uninstall the package"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run tests"
	@echo "  example     - Run example script"
	@echo "  benchmark   - Run benchmarks"
	@echo "  lint        - Check code style"
	@echo "  help        - Show this help message"

# Development workflow
dev: clean build test example

# Full installation workflow
full: clean build install-dev test example 