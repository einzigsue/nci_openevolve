// N-Body Simulation CUDA Kernel with Bulletproof Verification
// Clean implementation with matching GPU/CPU algorithms

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include "kernel.h"


// Simple but robust GPU kernel - we'll make CPU match this EXACTLY
__global__ void nbody_kernel(float4* pos_in, float4* vel_in, 
                            float4* pos_out, float4* vel_out, 
                            float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // Load this particle's current state
    float4 my_pos = pos_in[i];
    float4 my_vel = vel_in[i];
    
    // Calculate total force on this particle
    float3 total_force = {0.0f, 0.0f, 0.0f};
    
    for (int j = 0; j < n; j++) {
        if (i != j) {
            float4 other_pos = pos_in[j];
            
            // Calculate distance vector
            float dx = other_pos.x - my_pos.x;
            float dy = other_pos.y - my_pos.y;
            float dz = other_pos.z - my_pos.z;
            
            // Calculate distance squared with softening
            float dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING;
            
            // Calculate 1/r^3 using fast inverse square root
            float inv_dist = rsqrtf(dist_sq);
            float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
            
            // Calculate force magnitude: F = G * m1 * m2 / r^2
            // But we want acceleration: a = F/m1 = G * m2 / r^2
            float force_magnitude = G * other_pos.w * inv_dist_cubed;
            
            // Add force components
            total_force.x += force_magnitude * dx;
            total_force.y += force_magnitude * dy;
            total_force.z += force_magnitude * dz;
        }
    }
    
    // Update velocity: v_new = v_old + a * dt
    float4 new_vel = my_vel;
    new_vel.x += total_force.x * dt;
    new_vel.y += total_force.y * dt;
    new_vel.z += total_force.z * dt;
    
    // Update position: x_new = x_old + v_old * dt (using OLD velocity for stability)
    float4 new_pos = my_pos;
    new_pos.x += my_vel.x * dt;
    new_pos.y += my_vel.y * dt;
    new_pos.z += my_vel.z * dt;
    
    // Write results
    pos_out[i] = new_pos;
    vel_out[i] = new_vel;
}

void nbody_gpu_step(float4* d_pos_a, float4* d_vel_a, 
                   float4* d_pos_b, float4* d_vel_b, 
                   float dt, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
//    printf("    GPU kernel launch: grid=%d, block=%d, n=%d\n", grid_size, block_size, n);
    
    nbody_kernel<<<grid_size, block_size>>>(d_pos_a, d_vel_a, d_pos_b, d_vel_b, dt, n);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("    CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("    CUDA kernel execution error: %s\n", cudaGetErrorString(err));
    } 
}

// Performance benchmark
void run_benchmark(int n, int steps) {
    printf("\n=== PERFORMANCE BENCHMARK ===\n");
    printf("Particles: %d, Steps: %d\n", n, steps);
    
    size_t bytes = n * sizeof(float4);
    
    // Allocate memory
    float4* h_pos_a = (float4*)malloc(bytes);
    float4* h_vel_a = (float4*)malloc(bytes);
    
    float4 *d_pos_a, *d_vel_a, *d_pos_b, *d_vel_b;
    cudaMalloc(&d_pos_a, bytes);
    cudaMalloc(&d_vel_a, bytes);
    cudaMalloc(&d_pos_b, bytes);
    cudaMalloc(&d_vel_b, bytes);
    
    // Initialize
    srand(42);
    for (int i = 0; i < n; i++) {
        h_pos_a[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        h_pos_a[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        h_pos_a[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        h_pos_a[i].w = 1.0f;
        
        h_vel_a[i].x = 0.1f * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
        h_vel_a[i].y = 0.1f * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
        h_vel_a[i].z = 0.1f * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
        h_vel_a[i].w = 0.0f;
    }
    
    cudaMemcpy(d_pos_a, h_pos_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_a, h_vel_a, bytes, cudaMemcpyHostToDevice);
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    nbody_gpu_step(d_pos_a, d_vel_a, d_pos_b, d_vel_b, 0.01f, n);
    
    cudaEventRecord(start);
    
    // Benchmark loop
    for (int step = 0; step < steps; step++) {
        nbody_gpu_step(d_pos_a, d_vel_a, d_pos_b, d_vel_b, 0.01f, n);
        
        // Swap buffers
        float4* temp;
        temp = d_pos_a; d_pos_a = d_pos_b; d_pos_b = temp;
        temp = d_vel_a; d_vel_a = d_vel_b; d_vel_b = temp;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_per_step = elapsed_ms / steps;
    
    // Calculate metrics
    long long interactions = (long long)n * (n - 1);
    // the total flops per force calc can be measured differently
    long long total_flops = interactions * 15;
    float gflops = (total_flops / 1e9) / (avg_time_per_step / 1000.0f);
    
    printf("Total time: %.2f ms\n", elapsed_ms);
    printf("Time per step: %.2f ms\n", avg_time_per_step);
    printf("GFLOPS: %.2f\n", gflops);
    
    // Cleanup
    free(h_pos_a); free(h_vel_a);
    cudaFree(d_pos_a); cudaFree(d_vel_a); cudaFree(d_pos_b); cudaFree(d_vel_b);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

