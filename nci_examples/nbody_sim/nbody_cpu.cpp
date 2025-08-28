// N-Body Simulation CUDA Kernel with Bulletproof Verification
// Clean implementation with matching GPU/CPU algorithms

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include "kernel.h"


// CPU version that matches GPU kernel EXACTLY
void nbody_cpu_step(float4* pos_in, float4* vel_in, 
                   float4* pos_out, float4* vel_out, 
                   float dt, int n) {
    
    for (int i = 0; i < n; i++) {
        // Load this particle's current state (matches GPU)
        float4 my_pos = pos_in[i];
        float4 my_vel = vel_in[i];
        
        // Calculate total force on this particle (matches GPU)
        float3 total_force = {0.0f, 0.0f, 0.0f};
        
        for (int j = 0; j < n; j++) {
            if (i != j) {
                float4 other_pos = pos_in[j];
                
                // Calculate distance vector (matches GPU)
                float dx = other_pos.x - my_pos.x;
                float dy = other_pos.y - my_pos.y;
                float dz = other_pos.z - my_pos.z;
                
                // Calculate distance squared with softening (matches GPU)
                float dist_sq = dx*dx + dy*dy + dz*dz + SOFTENING;
                
                // Calculate 1/r^3 - use regular sqrt to match rsqrtf as closely as possible
                float inv_dist = 1.0f / sqrtf(dist_sq);
                float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
                
                // Calculate force magnitude (matches GPU)
                float force_magnitude = G * other_pos.w * inv_dist_cubed;
                
                // Add force components (matches GPU)
                total_force.x += force_magnitude * dx;
                total_force.y += force_magnitude * dy;
                total_force.z += force_magnitude * dz;
            }
        }
        // Update velocity: v_new = v_old + a * dt (matches GPU)
        float4 new_vel = my_vel;
        new_vel.x += total_force.x * dt;
        new_vel.y += total_force.y * dt;
        new_vel.z += total_force.z * dt;
        
        // Update position: x_new = x_old + v_old * dt (matches GPU - using OLD velocity)
        float4 new_pos = my_pos;
        new_pos.x += my_vel.x * dt;
        new_pos.y += my_vel.y * dt;
        new_pos.z += my_vel.z * dt;
        
        // Write results (matches GPU)
        pos_out[i] = new_pos;
        vel_out[i] = new_vel;
    }
}



// Initialize particles with known, simple configuration
void init_simple_system(float4* pos, float4* vel, int n) {
    // Use a deterministic, simple setup for verification
    for (int i = 0; i < n; i++) {
        // Simple grid-like positions
        pos[i].x = (float)(i % 8) - 4.0f;          // -4 to 3
        pos[i].y = (float)((i / 8) % 8) - 4.0f;    // -4 to 3  
        pos[i].z = (float)(i / 64) - 2.0f;         // -2 to ?
        pos[i].w = 1.0f;  // Unit mass
        
        // Zero initial velocities for predictable behavior
        vel[i].x = 0.0f;
        vel[i].y = 0.0f;
        vel[i].z = 0.0f;
        vel[i].w = 0.0f;
    }
}

// Precise comparison function
bool verify_results(float4* pos1, float4* vel1, float4* pos2, float4* vel2, int n) {
    float max_pos_diff = 0.0f;
    float max_vel_diff = 0.0f;
    float sum_pos_diff = 0.0f;
    float sum_vel_diff = 0.0f;
    
    //printf("Detailed comparison (first 5 particles):\n");
    
    for (int i = 0; i < n; i++) {
        // Position differences
        float pos_diff_x = fabsf(pos1[i].x - pos2[i].x);
        float pos_diff_y = fabsf(pos1[i].y - pos2[i].y);
        float pos_diff_z = fabsf(pos1[i].z - pos2[i].z);
        float pos_diff_total = sqrtf(pos_diff_x*pos_diff_x + pos_diff_y*pos_diff_y + pos_diff_z*pos_diff_z);
        
        // Velocity differences  
        float vel_diff_x = fabsf(vel1[i].x - vel2[i].x);
        float vel_diff_y = fabsf(vel1[i].y - vel2[i].y);
        float vel_diff_z = fabsf(vel1[i].z - vel2[i].z);
        float vel_diff_total = sqrtf(vel_diff_x*vel_diff_x + vel_diff_y*vel_diff_y + vel_diff_z*vel_diff_z);
        
        sum_pos_diff += pos_diff_total;
        sum_vel_diff += vel_diff_total;
        
        if (pos_diff_total > max_pos_diff) max_pos_diff = pos_diff_total;
        if (vel_diff_total > max_vel_diff) max_vel_diff = vel_diff_total;
        
    }
    
    float avg_pos_diff = sum_pos_diff / n;
    float avg_vel_diff = sum_vel_diff / n;
    
    printf("\nStatistics:\n");
    printf("Position - Max diff: %.2e, Avg diff: %.2e\n", max_pos_diff, avg_pos_diff);
    printf("Velocity - Max diff: %.2e, Avg diff: %.2e\n", max_vel_diff, avg_vel_diff);
    
    // Be more lenient - rsqrtf vs sqrtf will cause small differences
    bool pos_ok = (max_pos_diff < 1e-3f);  // 0.001 tolerance
    bool vel_ok = (max_vel_diff < 1e-3f);  // 0.001 tolerance
    
    if (pos_ok && vel_ok) {
        printf("VERIFICATION PASSED: Differences are within acceptable tolerance\n");
        printf("  (Small differences are expected due to rsqrtf vs sqrtf precision)\n");
        return true;
    } else {
        printf("VERIFICATION FAILED: Differences exceed tolerance\n");
        if (!pos_ok) printf("  Position differences too large (max: %.6f > 0.001000)\n", max_pos_diff);
        if (!vel_ok) printf("  Velocity differences too large (max: %.6f > 0.001000)\n", max_vel_diff);
        return false;
    }
}

// Run verification test with debugging
void run_verification(int n, int steps) {
    printf("\n=== VERIFICATION TEST ===\n");
    printf("Particles: %d, Steps: %d\n", n, steps);
    
    size_t bytes = n * sizeof(float4);
    
    // Allocate CPU memory  
    float4* h_pos_a = (float4*)malloc(bytes);
    float4* h_vel_a = (float4*)malloc(bytes);
    float4* h_pos_b = (float4*)malloc(bytes);  
    float4* h_vel_b = (float4*)malloc(bytes);
    
    float4* h_pos_gpu = (float4*)malloc(bytes);
    float4* h_vel_gpu = (float4*)malloc(bytes);
    
    // Initialize with simple, deterministic system
    init_simple_system(h_pos_a, h_vel_a, n);
    
    
    // Copy initial state for CPU test
    memcpy(h_pos_b, h_pos_a, bytes);
    memcpy(h_vel_b, h_vel_a, bytes);
    
    // Allocate GPU memory
    float4 *d_pos_a, *d_vel_a, *d_pos_b, *d_vel_b;
    cudaError_t err;
    
    err = cudaMalloc(&d_pos_a, bytes); if (err != cudaSuccess) printf("CUDA Error 1: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_vel_a, bytes); if (err != cudaSuccess) printf("CUDA Error 2: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_pos_b, bytes); if (err != cudaSuccess) printf("CUDA Error 3: %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_vel_b, bytes); if (err != cudaSuccess) printf("CUDA Error 4: %s\n", cudaGetErrorString(err));
    
    // Initialize GPU buffers to zero first
    err = cudaMemset(d_pos_a, 0, bytes); if (err != cudaSuccess) printf("CUDA Error 5: %s\n", cudaGetErrorString(err));
    err = cudaMemset(d_vel_a, 0, bytes); if (err != cudaSuccess) printf("CUDA Error 6: %s\n", cudaGetErrorString(err));
    err = cudaMemset(d_pos_b, 0, bytes); if (err != cudaSuccess) printf("CUDA Error 7: %s\n", cudaGetErrorString(err));
    err = cudaMemset(d_vel_b, 0, bytes); if (err != cudaSuccess) printf("CUDA Error 8: %s\n", cudaGetErrorString(err));
    
    // Copy initial state to GPU
    err = cudaMemcpy(d_pos_a, h_pos_a, bytes, cudaMemcpyHostToDevice); 
    if (err != cudaSuccess) printf("CUDA Error 9: %s\n", cudaGetErrorString(err));
    err = cudaMemcpy(d_vel_a, h_vel_a, bytes, cudaMemcpyHostToDevice); 
    if (err != cudaSuccess) printf("CUDA Error 10: %s\n", cudaGetErrorString(err));
    
    // Verify data was copied correctly
    cudaMemcpy(h_pos_gpu, d_pos_a, bytes, cudaMemcpyDeviceToHost);
    
    float dt = 0.01f;
    
    //printf("Running simulation for %d steps...\n", steps);
    
    // Keep track of current buffers (start with A as input, B as output)
    float4 *d_pos_current = d_pos_a, *d_pos_next = d_pos_b;
    float4 *d_vel_current = d_vel_a, *d_vel_next = d_vel_b;
    float4 *h_pos_current = h_pos_a, *h_pos_next = h_pos_b;
    float4 *h_vel_current = h_vel_a, *h_vel_next = h_vel_b;
    
    // Run both versions for specified steps
    for (int step = 0; step < steps; step++) {
        //printf("  Step %d...\n", step + 1);
        
        // CPU step 
        nbody_cpu_step(h_pos_current, h_vel_current, h_pos_next, h_vel_next, dt, n);
        
        // GPU step  
        nbody_gpu_step(d_pos_current, d_vel_current, d_pos_next, d_vel_next, dt, n);
        
        // Check for CUDA errors
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error after step %d: %s\n", step + 1, cudaGetErrorString(err));
        }
        
        // Swap buffers for next iteration
        float4* temp;
        temp = h_pos_current; h_pos_current = h_pos_next; h_pos_next = temp;
        temp = h_vel_current; h_vel_current = h_vel_next; h_vel_next = temp;
        temp = d_pos_current; d_pos_current = d_pos_next; d_pos_next = temp;
        temp = d_vel_current; d_vel_current = d_vel_next; d_vel_next = temp;
    }
    
    // Copy final GPU results back (from current, which is the final state)
    cudaMemcpy(h_pos_gpu, d_pos_current, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_gpu, d_vel_current, bytes, cudaMemcpyDeviceToHost);
    
    // Compare final results
    bool success = verify_results(h_pos_gpu, h_vel_gpu, h_pos_current, h_vel_current, n);
    
    // Cleanup
    free(h_pos_a); free(h_vel_a); free(h_pos_b); free(h_vel_b); free(h_pos_gpu); free(h_vel_gpu);
    cudaFree(d_pos_a); cudaFree(d_vel_a); cudaFree(d_pos_b); cudaFree(d_vel_b);
    
    if (success) {
        printf("GPU and CPU implementations are verified to be equivalent!\n");
    } else {
        printf("There may be a bug in one of the implementations.\n");
    }
}

