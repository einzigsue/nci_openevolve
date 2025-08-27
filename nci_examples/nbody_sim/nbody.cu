// N-Body Simulation CUDA Kernel
// Tests: Heavy compute, memory bandwidth, transcendental functions

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

struct Particle {
    float4 pos;  // x, y, z, mass (using w component for mass)
    float4 vel;  // vx, vy, vz, (w unused)
};

// Softening factor to prevent singularities when particles get too close
#define SOFTENING 1e-9f
#define G 6.67430e-11f  // Gravitational constant (can be scaled for simulation)

// Single kernel N-body - O(N²) complexity with position update
__global__ void nbody_single_kernel(float4* pos, float4* vel, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float3 force = {0.0f, 0.0f, 0.0f};
    float4 myPos = pos[i];
    float4 myVel = vel[i];
    
    for (int j = 0; j < n; j++) {
        if (i != j) {  // Don't calculate force on self
            float4 otherPos = pos[j];
            
            float3 r;
            r.x = otherPos.x - myPos.x;
            r.y = otherPos.y - myPos.y;
            r.z = otherPos.z - myPos.z;
            
            float distSqr = r.x*r.x + r.y*r.y + r.z*r.z + SOFTENING;
            float invDist = rsqrtf(distSqr);  // Fast inverse square root
            float invDist3 = invDist * invDist * invDist;
            
            // F = G * m1 * m2 / r²
            // But we calculate acceleration directly: a = F/m1 = G * m2 / r²
            float forceMag = G * otherPos.w * invDist3;
            
            force.x += forceMag * r.x;
            force.y += forceMag * r.y;
            force.z += forceMag * r.z;
        }
    }
    
    // Update velocity: v = v + a*dt
    myVel.x += force.x * dt;
    myVel.y += force.y * dt;
    myVel.z += force.z * dt;
    
    // Update position: pos = pos + v*dt
    myPos.x += myVel.x * dt;
    myPos.y += myVel.y * dt;
    myPos.z += myVel.z * dt;
    // Mass (myPos.w) stays constant
    
    // Write back to global memory
    vel[i] = myVel;
    pos[i] = myPos;
}

// Single kernel host function
void nbody_step_single(float4* d_pos, float4* d_vel, float dt, int n) {
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    nbody_single_kernel<<<gridSize, blockSize>>>(d_pos, d_vel, dt, n);
}

// Performance metrics calculation
void calculate_performance_metrics(int n, float time_ms) {
    // Each particle interacts with (n-1) others
    long long interactions = (long long)n * (n - 1);
    
    // Each interaction involves:
    // - 3 subtractions (distance vector)
    // - 3 multiplications + 2 additions (distance squared)
    // - 1 rsqrt, 2 multiplications (inv distance cubed)  
    // - 1 multiplication (force magnitude)
    // - 3 multiplications + 3 additions (force components)
    // Total: ~15 FLOPs per interaction
    
    long long total_flops = interactions * 15;
    float gflops = (total_flops / 1e9) / (time_ms / 1000.0f);
    
    // Memory access: each particle reads all other particles
    // 4 floats per particle * n particles read n times
    long long bytes_read = (long long)n * n * 4 * sizeof(float);
    float bandwidth_gb_s = (bytes_read / 1e9) / (time_ms / 1000.0f);
    
    printf("N-Body Performance Metrics:\n");
    printf("Particles: %d\n", n);
    printf("Interactions: %lld\n", interactions);
    printf("Time: %.2f ms\n", time_ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("Arithmetic Intensity: %.2f FLOP/byte\n", 
           (float)total_flops / bytes_read);
}

// Initialize particles in a random distribution
void initialize_particles(float4* pos, float4* vel, int n) {
    srand(42);  // Fixed seed for reproducible results
    
    for (int i = 0; i < n; i++) {
        // Random positions in a cube [-1, 1]
        pos[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        pos[i].w = 1.0f;  // Unit mass
        
        // Small random velocities
        vel[i].x = 0.1f * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
        vel[i].y = 0.1f * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
        vel[i].z = 0.1f * (2.0f * (rand() / (float)RAND_MAX) - 1.0f);
        vel[i].w = 0.0f;  // Unused
    }
}

// Main function - the missing piece!
int main(int argc, char** argv) {
    // Default parameters
    int n = 4096;          // Number of particles
    int num_steps = 100;   // Number of simulation steps
    float dt = 0.01f;      // Timestep
    
    // Parse command line arguments
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) num_steps = atoi(argv[2]);
    
    printf("N-Body Simulation Benchmark\n");
    printf("===========================\n");
    printf("Particles: %d\n", n);
    printf("Steps: %d\n", num_steps);
    printf("\n");
    
    // Allocate host memory
    size_t bytes = n * sizeof(float4);
    float4* h_pos = (float4*)malloc(bytes);
    float4* h_vel = (float4*)malloc(bytes);
    
    // Initialize particles
    initialize_particles(h_pos, h_vel, n);
    
    // Allocate device memory
    float4* d_pos;
    float4* d_vel;
    cudaMalloc(&d_pos, bytes);
    cudaMalloc(&d_vel, bytes);
    
    // Copy to device
    cudaMemcpy(d_pos, h_pos, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel, h_vel, bytes, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up GPU
    nbody_step_single(d_pos, d_vel, dt, n);
    cudaDeviceSynchronize();
    
    // Start timing
    cudaEventRecord(start);
    
    // Run simulation
    for (int step = 0; step < num_steps; step++) {
        nbody_step_single(d_pos, d_vel, dt, n);
    }
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    float avg_time_per_step = elapsed_ms / num_steps;
    
    // Calculate and print performance metrics
    calculate_performance_metrics(n, avg_time_per_step);
    
    // Optional: Copy results back and save final state
    if (n <= 10) {  // Only print for small systems
        cudaMemcpy(h_pos, d_pos, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vel, d_vel, bytes, cudaMemcpyDeviceToHost);
        
        printf("\nFinal particle states:\n");
        for (int i = 0; i < n; i++) {
            printf("Particle %d: pos(%.3f, %.3f, %.3f) vel(%.3f, %.3f, %.3f)\n",
                   i, h_pos[i].x, h_pos[i].y, h_pos[i].z,
                   h_vel[i].x, h_vel[i].y, h_vel[i].z);
        }
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_pos);
    cudaFree(d_vel);
    free(h_pos);
    free(h_vel);
    
    printf("\nBenchmark completed successfully!\n");
    return 0;
}
