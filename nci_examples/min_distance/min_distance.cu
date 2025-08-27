// Closest Pair of 3D Particles CUDA Kernel
// Tests: Memory bandwidth, reductions, atomic operations, divergent branching

#include <cuda_runtime.h>
#include <chrono>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Structure for 3D particle
struct Particle {
    float x, y, z;
    int id;  // Particle identifier
};

// Result structure for closest pair
struct ClosestPair {
    float distance;
    int particle1;
    int particle2;
};

// Naive approach - each thread checks one particle against all others
__global__ void find_closest_pair_naive(Particle* particles, int n, 
                                       float* min_distances, int* pair_indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    float min_dist = FLT_MAX;
    int closest_particle = -1;
    
    Particle pi = particles[i];
    
    // Check against all other particles
    for (int j = 0; j < n; j++) {
        if (i != j) {
            Particle pj = particles[j];
            
            // Calculate squared distance (avoid sqrt for performance)
            float dx = pi.x - pj.x;
            float dy = pi.y - pj.y;
            float dz = pi.z - pj.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq < min_dist) {
                min_dist = dist_sq;
                closest_particle = j;
            }
        }
    }
    
    // Store results (each thread finds its closest neighbor)
    min_distances[i] = sqrtf(min_dist);
    pair_indices[i] = closest_particle;
}

// CPU reference implementation for validation
ClosestPair find_closest_pair_cpu(Particle* particles, int n) {
    ClosestPair result;
    result.distance = FLT_MAX;
    result.particle1 = -1;
    result.particle2 = -1;
    
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dx = particles[i].x - particles[j].x;
            float dy = particles[i].y - particles[j].y;
            float dz = particles[i].z - particles[j].z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            
            if (dist < result.distance) {
                result.distance = dist;
                result.particle1 = i;
                result.particle2 = j;
            }
        }
    }
    
    return result;
}

// Initialize random particles
void initialize_particles(Particle* particles, int n) {
    srand(42);  // Fixed seed for reproducible results
    
    for (int i = 0; i < n; i++) {
        particles[i].x = 10.0f * (rand() / (float)RAND_MAX);
        particles[i].y = 10.0f * (rand() / (float)RAND_MAX);
        particles[i].z = 10.0f * (rand() / (float)RAND_MAX);
        particles[i].id = i;
    }
}

// Performance metrics calculation
void calculate_performance_metrics(int n, float time_ms, const char* method) {
    // Each pair is checked once: n*(n-1)/2 distance calculations
    long long comparisons = (long long)n * (n - 1) / 2;
    
    // Each comparison: 3 subtractions, 3 multiplications, 2 additions, 1 sqrt
    // Total: ~9 FLOPs per comparison
    long long total_flops = comparisons * 9;
    float gflops = (total_flops / 1e9) / (time_ms / 1000.0f);
    
    // Memory access: reading particle data
    long long bytes_read = (long long)n * n * sizeof(Particle);
    float bandwidth_gb_s = (bytes_read / 1e9) / (time_ms / 1000.0f);
    
    printf("\n%s Performance:\n", method);
    printf("Particles: %d\n", n);
    printf("Comparisons: %lld\n", comparisons);
    printf("Time: %.2f ms\n", time_ms);
    printf("GFLOPS: %.2f\n", gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    printf("Arithmetic Intensity: %.2f FLOP/byte\n", gflops / bandwidth_gb_s);
}

int main(int argc, char** argv) {
    int n = 4096;  // Default number of particles
    if (argc > 1) n = atoi(argv[1]);
    
    printf("Closest Pair of Particles Benchmark\n");
    printf("===================================\n");
    printf("Particles: %d\n\n", n);
    
    // Allocate host memory
    size_t particles_bytes = n * sizeof(Particle);
    Particle* h_particles = (Particle*)malloc(particles_bytes);
    
    // Initialize particles
    initialize_particles(h_particles, n);
    
    // Allocate device memory
    Particle* d_particles;
    cudaMalloc(&d_particles, particles_bytes);
    cudaMemcpy(d_particles, h_particles, particles_bytes, cudaMemcpyHostToDevice);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Method 1: Naive approach
    {
        float* d_min_distances;
        int* d_pair_indices;
        cudaMalloc(&d_min_distances, n * sizeof(float));
        cudaMalloc(&d_pair_indices, n * sizeof(int));
        
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        
        cudaEventRecord(start);
        find_closest_pair_naive<<<gridSize, blockSize>>>(d_particles, n, 
                                                        d_min_distances, d_pair_indices);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        calculate_performance_metrics(n, elapsed_ms, "Naive Method");
        
        cudaFree(d_min_distances);
        cudaFree(d_pair_indices);
    }
    
    
    // CPU reference for validation (only for small N)
    if (n <= 1000) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        ClosestPair cpu_result = find_closest_pair_cpu(h_particles, n);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        
        float cpu_time_ms = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();
        
        printf("\nCPU Reference:\n");
        printf("Time: %.2f ms\n", cpu_time_ms);
        printf("Closest pair: particles %d and %d, distance: %.6f\n", 
               cpu_result.particle1, cpu_result.particle2, cpu_result.distance);
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_particles);
    free(h_particles);
    
    printf("\nBenchmark completed!\n");
    return 0;
}
