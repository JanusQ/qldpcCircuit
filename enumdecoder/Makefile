# Makefile for QLDPC Sample Decoder

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++11 -O3 -arch=sm_50 -lcurand
CXX_FLAGS = -std=c++11 -O3 -Wall

# Source files
CUDA_SRC = sample.cu
CSC_CUDA_SRC = csc_cuda.cu
CPP_SRC = sample_decoder.cpp
ORIG_SRC = qldpc_decoder.cpp
CSC_SRC = qldpc_decoder_csc.cpp
TEST_SRC = test_comparison.cpp
PERF_SRC = performance_test_csc.cpp

# Object files
CUDA_OBJ = sample.o
CSC_CUDA_OBJ = csc_cuda.o
CPP_OBJ = sample_decoder.o
ORIG_OBJ = qldpc_decoder.o
CSC_OBJ = qldpc_decoder_csc.o
TEST_OBJ = test_comparison.o
PERF_OBJ = performance_test_csc.o

# Executable names
SAMPLE_EXEC = sample_decoder
ORIG_EXEC = qldpc_decoder
CSC_EXEC = qldpc_decoder_csc
TEST_EXEC = test_comparison
PERF_EXEC = performance_test_csc

# Default target
all: $(SAMPLE_EXEC) $(ORIG_EXEC) $(CSC_EXEC)

# Build all including test
all_with_test: $(SAMPLE_EXEC) $(ORIG_EXEC) $(CSC_EXEC) $(TEST_EXEC) $(PERF_EXEC)

# Build sample decoder with CUDA
$(SAMPLE_EXEC): $(CUDA_OBJ) $(CPP_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Build original decoder
$(ORIG_EXEC): $(ORIG_OBJ)
	g++ $(CXX_FLAGS) -o $@ $^

# Build CSC-optimized decoder
$(CSC_EXEC): $(CSC_CUDA_OBJ) $(CSC_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Build test comparison (requires both decoders)
$(TEST_EXEC): $(CUDA_OBJ) $(TEST_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Build performance test
$(PERF_EXEC): $(CSC_CUDA_OBJ) $(PERF_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Shortcut for test comparison
test_comparison: $(TEST_EXEC)

# Shortcut for performance test
performance_test: $(PERF_EXEC)

# Compile CUDA source
$(CUDA_OBJ): $(CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile CSC CUDA source
$(CSC_CUDA_OBJ): $(CSC_CUDA_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source for sample decoder
$(CPP_OBJ): $(CPP_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile original C++ source
$(ORIG_OBJ): $(ORIG_SRC)
	g++ $(CXX_FLAGS) -c $< -o $@

# Compile CSC-optimized C++ source
$(CSC_OBJ): $(CSC_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile test comparison source
$(TEST_OBJ): $(TEST_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile performance test source
$(PERF_OBJ): $(PERF_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f *.o $(SAMPLE_EXEC) $(ORIG_EXEC) $(CSC_EXEC) $(TEST_EXEC) $(PERF_EXEC)

# Run sample decoder
run_sample: $(SAMPLE_EXEC)
	./$(SAMPLE_EXEC)

# Run original decoder
run_orig: $(ORIG_EXEC)
	./$(ORIG_EXEC)

# Run CSC-optimized decoder
run_csc: $(CSC_EXEC)
	./$(CSC_EXEC)

# Run performance comparison
run_test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Run CSC performance test
run_perf: $(PERF_EXEC)
	./$(PERF_EXEC)

# Build Python package
python_pkg:
	python setup.py build_ext --inplace

# Run Python example
run_python: python_pkg
	python example_usage.py

# Check CUDA installation
check_cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@nvidia-smi

# Help
help:
	@echo "Available targets:"
	@echo "  all         - Build all decoders (sample, original, CSC)"
	@echo "  sample_decoder - Build sample decoder with CUDA"
	@echo "  qldpc_decoder  - Build original decoder"
	@echo "  qldpc_decoder_csc - Build CSC-optimized decoder"
	@echo "  test_comparison - Build performance comparison tool"
	@echo "  performance_test_csc - Build CSC performance test"
	@echo "  python_pkg  - Build Python package"
	@echo "  run_sample  - Run sample decoder"
	@echo "  run_orig    - Run original decoder"
	@echo "  run_csc     - Run CSC-optimized decoder"
	@echo "  run_test    - Run performance comparison"
	@echo "  run_perf    - Run CSC performance test"
	@echo "  run_python  - Run Python example"
	@echo "  check_cuda  - Check CUDA installation"
	@echo "  clean       - Clean build files"
	@echo "  help        - Show this help message"

.PHONY: all all_with_test clean run_sample run_orig run_csc run_test run_perf run_python check_cuda help test_comparison performance_test python_pkg 