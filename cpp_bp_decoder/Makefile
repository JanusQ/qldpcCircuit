CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic
DEBUG_FLAGS = -g -DDEBUG
RELEASE_FLAGS = -DNDEBUG

# Source files
SOURCES = belief_propagation.cpp example.cpp
HEADERS = belief_propagation.hpp
OBJECTS = $(SOURCES:.cpp=.o)

# Target executable
TARGET = bp_decoder_example

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

# Build object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: clean $(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the example
run: $(TARGET)
	./$(TARGET)

# Create a test with valgrind (if available)
test: $(TARGET)
	@if command -v valgrind >/dev/null 2>&1; then \
		echo "Running with valgrind..."; \
		valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET); \
	else \
		echo "Valgrind not found, running normally..."; \
		./$(TARGET); \
	fi

# Install dependencies (placeholder)
install-deps:
	@echo "No external dependencies required for this C++17 implementation"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the executable (default)"
	@echo "  debug      - Build with debug flags"
	@echo "  release    - Build with release optimizations"
	@echo "  clean      - Remove build artifacts"
	@echo "  run        - Build and run the example"
	@echo "  test       - Run with memory checking (valgrind)"
	@echo "  help       - Show this help message"

# Phony targets
.PHONY: all debug release clean run test install-deps help 