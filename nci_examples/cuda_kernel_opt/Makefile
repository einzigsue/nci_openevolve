# Makefile for CUDA matrix transpose

# Compiler and flags
NVCC = nvcc
ARCH = sm_70
CFLAGS = --gpu-architecture=$(ARCH) -O3
TARGET = transpose
SOURCE = transpose.cu

# Default target
all: $(TARGET)

# Compile the CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SOURCE)

# Clean up compiled files
clean:
	rm -f $(TARGET)

# Run the program with default parameters
run: $(TARGET)
	./$(TARGET) 1024 1024

# Run with small test case
test: $(TARGET)
	./$(TARGET) 4 3

# Run with various sizes for benchmarking
benchmark: $(TARGET)
	@echo "Running benchmarks..."
	./$(TARGET) 1024 1024
	./$(TARGET) 2048 2048
	./$(TARGET) 4096 4096
	./$(TARGET) 8192 8192

# Profile with nsight compute (if available)
profile: $(TARGET)
	ncu --set full ./$(TARGET) 2048 2048

# Debug build with device debug info
debug: CFLAGS += -g -G
debug: $(TARGET)

# Show help
help:
	@echo "Available targets:"
	@echo "  all       - Build the program (default)"
	@echo "  clean     - Remove compiled files"
	@echo "  run       - Build and run with 1024x1024 matrix"
	@echo "  test      - Build and run with small 4x3 matrix"
	@echo "  benchmark - Run multiple sizes for performance testing"
	@echo "  profile   - Run with nsight compute profiler"
	@echo "  debug     - Build with debug symbols"
	@echo "  help      - Show this help message"

# Declare phony targets
.PHONY: all clean run test benchmark profile debug help
