.PHONY: all build clean test test-logical test-scaling install

all: build

build:
	python setup.py build_ext --inplace

install:
	pip install -e .

test-logical: build
	python test_logical_error_rate.py

test-scaling: build
	python test_scaling_analysis.py

clean:
	rm -rf build/
	rm -rf *.so
	rm -rf __pycache__/
	rm -rf *.egg-info/
	rm -rf *.png
	rm -rf *.json
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete 