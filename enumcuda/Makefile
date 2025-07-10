# Makefile for CUDA QLDPC Decoder
# High-performance CUDA implementation

# Compiler settings
NVCC = nvcc
CXX = g++
CC = gcc

# CUDA settings
CUDA_PATH = /usr/local/cuda
CUDA_ARCH = -arch=compute_80  # Adjust based on your GPU
CUDA_FLAGS = -O3 -std=c++11 $(CUDA_ARCH)

# Compiler flags
CXXFLAGS = -O3 -std=c++11 -Wall -Wextra
INCLUDES = -I$(CUDA_PATH)/include -I.
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand

# Source files
CUDA_SOURCES = cuda_decoder.cpp
CUDA_KERNELS = cuda_kernels.cu
OBJECTS = $(CUDA_SOURCES:.cpp=.o) cuda_kernels.o

# Target executable
TARGET = cuda_decoder

# Default target
all: $(TARGET)

# Build the CUDA decoder
$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Compile CUDA source files
%.o: %.cpp
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA kernel files
cuda_kernels.o: cuda_kernels.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@



# Performance test
test_performance: $(TARGET)
	./$(TARGET)

# Benchmark different batch sizes
benchmark: $(TARGET)
	@echo "Benchmarking CUDA decoder with different batch sizes..."
	@echo "Batch size 1000:"
	@CUDA_BATCH_SIZE=1000 ./$(TARGET) 2>/dev/null | grep "CUDA Decoder initialized"
	@echo "Batch size 10000:"
	@CUDA_BATCH_SIZE=10000 ./$(TARGET) 2>/dev/null | grep "CUDA Decoder initialized"
	@echo "Batch size 50000:"
	@CUDA_BATCH_SIZE=50000 ./$(TARGET) 2>/dev/null | grep "CUDA Decoder initialized"

# Memory usage test
memory_test: $(TARGET)
	@echo "Testing memory usage..."
	@valgrind --tool=massif --massif-out-file=memory_profile.out ./$(TARGET)
	@echo "Memory profile saved to memory_profile.out"

# GPU memory check
gpu_memory: $(TARGET)
	@echo "GPU memory usage before:"
	@nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
	@./$(TARGET) > /dev/null
	@echo "GPU memory usage after:"
	@nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET) *.o

# Clean all generated files
distclean: clean
	rm -f memory_profile.out *.log

# Install dependencies (Ubuntu/Debian)
install_deps:
	sudo apt-get update
	sudo apt-get install -y nvidia-cuda-toolkit nvidia-cuda-dev
	sudo apt-get install -y build-essential

# Check CUDA installation
check_cuda:
	@echo "Checking CUDA installation..."
	@nvcc --version
	@nvidia-smi
	@echo "CUDA_PATH: $(CUDA_PATH)"
	@ls -la $(CUDA_PATH)/lib64/libcudart.so

# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build the CUDA decoder"
	@echo "  test_performance - Run performance test"
	@echo "  benchmark        - Benchmark different batch sizes"
	@echo "  memory_test      - Test memory usage with valgrind"
	@echo "  gpu_memory       - Check GPU memory usage"
	@echo "  clean            - Remove build files"
	@echo "  distclean        - Remove all generated files"
	@echo "  install_deps     - Install dependencies (Ubuntu/Debian)"
	@echo "  check_cuda       - Check CUDA installation"
	@echo "  help             - Show this help"

# Phony targets
.PHONY: all test_performance benchmark memory_test gpu_memory clean distclean install_deps check_cuda help 