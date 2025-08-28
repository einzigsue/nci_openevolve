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

