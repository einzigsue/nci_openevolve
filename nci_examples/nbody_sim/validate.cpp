// N-Body Simulation CUDA Kernel with Bulletproof Verification
// Clean implementation with matching GPU/CPU algorithms

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include "kernel.h"

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 64;      // Smaller default for debugging
    int steps = (argc > 2) ? atoi(argv[2]) : 100;   // Just 1 step for debugging
    
    printf("N-Body Simulation with Clean Verification\n");
    printf("==========================================\n");
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using CUDA device: %s\n", deviceProp.name);
    
        run_verification(std::min(n, 64), std::min(steps, 5));  // Very small for debugging
    
    
    return 0;
}

