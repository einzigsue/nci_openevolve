// N-Body Simulation CUDA Kernel with Bulletproof Verification
// Clean implementation with matching GPU/CPU algorithms

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

// Constants
#define SOFTENING 1e-9f
#define G 6.67430e-11f

// Simple but robust GPU kernel - we'll make CPU match this EXACTLY
__global__ void nbody_kernel(float4* pos_in, float4* vel_in, 
                            float4* pos_out, float4* vel_out, 
                            float dt, int n); 

void nbody_cpu_step(float4* pos_in, float4* vel_in, 
                   float4* pos_out, float4* vel_out, 
                   float dt, int n); 

// Wrapper for GPU step using double buffering with debugging
void nbody_gpu_step(float4* d_pos_a, float4* d_vel_a, 
                   float4* d_pos_b, float4* d_vel_b, 
                   float dt, int n);

// Initialize particles with known, simple configuration
void init_simple_system(float4* pos, float4* vel, int n);

// Precise comparison function
bool verify_results(float4* pos1, float4* vel1, float4* pos2, float4* vel2, int n); 

// Run verification test with debugging
void run_verification(int n, int steps); 

// Performance benchmark
void run_benchmark(int n, int steps); 

