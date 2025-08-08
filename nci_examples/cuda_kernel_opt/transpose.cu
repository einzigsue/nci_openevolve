#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include <iomanip>
#define BLOCK_DIM 8
__global__ void matrixTranspose(float* out, const float* in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx  = y * width + x;
        int out_idx = x * height + y;
        out[out_idx] = in[in_idx];
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <nrows> <ncols>\n";
        return 1;
    }
    
    int height = std::atoi(argv[1]);  // nrows
    int width = std::atoi(argv[2]);   // ncols
    
    if (height <= 0 || width <= 0) {
        std::cerr << "Error: nrows and ncols must be positive integers\n";
        return 1;
    }
    
    int numElements = width * height;
    size_t size = numElements * sizeof(float);
    
    std::cout << "Matrix size: " << height << " x " << width << " (" << numElements << " elements)\n";
    
    // Allocate host arrays
    float* h_in  = new float[numElements];
    float* h_out = new float[numElements];
    
    // Initialize input with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    for (int i = 0; i < numElements; ++i) {
        h_in[i] = dis(gen);
    }
    
    // Print input matrix (only if small enough)
    if (height <= 10 && width <= 10) {
        std::cout << "Input matrix (" << height << "x" << width << "):\n";
        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                std::cout << std::fixed << std::setprecision(2) 
                         << h_in[row * width + col] << " ";
            }
            std::cout << "\n";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Matrix too large to display\n\n";
    }
    
    // Allocate device arrays
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemset(d_in, 0.0, size);
    cudaMemset(d_out, 0.0, size);
    
    // Copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    
    std::cout << "Grid dimensions: " << gridDim.x << " x " << gridDim.y 
              << " = " << gridDim.x * gridDim.y << " blocks\n";
    std::cout << "Block dimensions: " << blockDim.x << " x " << blockDim.y 
              << " = " << blockDim.x * blockDim.y << " threads per block\n";
    std::cout << "Total threads: " << gridDim.x * gridDim.y * blockDim.x * blockDim.y << "\n\n";
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up run
    for(int i = 0; i < 10; ++i){
    matrixTranspose<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceSynchronize();
    
    // Timed run
    cudaEventRecord(start, 0);
    matrixTranspose<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Kernel execution time: " << milliseconds << " ms\n";
    
    // Calculate effective memory bandwidth (read input + write output)
    size_t totalBytes = 2 * width * height * sizeof(float);
    float seconds = milliseconds / 1000.0f;
    float bandwidth = totalBytes / seconds / (1024.0f * 1024.0f * 1024.0f); // GB/s
    
    std::cout << "Total data moved: " << totalBytes / (1024.0f * 1024.0f) << " MB\n";
    std::cout << "Effective memory bandwidth: " << bandwidth << " GB/s\n";
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    
    // Print result (only if small enough)
    if (height <= 10 && width <= 10) {
        std::cout << "\nOutput (transposed) matrix (" << width << "x" << height << "):\n";
        for (int row = 0; row < width; ++row) {
            for (int col = 0; col < height; ++col) {
                std::cout << std::fixed << std::setprecision(2) 
                         << h_out[row * height + col] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "\nTransposed matrix too large to display\n";
    }
    
    // Verify correctness for small matrices
    if (height <= 1000 && width <= 1000) {
        bool correct = true;
        for (int row = 0; row < height && correct; ++row) {
            for (int col = 0; col < width && correct; ++col) {
                float input_val = h_in[row * width + col];
                float output_val = h_out[col * height + row];
                if (std::abs(input_val - output_val) > 1e-5) {
                    correct = false;
                    std::cout << "Mismatch at (" << row << "," << col << "): "
                             << "input=" << input_val << ", output=" << output_val << "\n";
                }
            }
        }
        std::cout << "Transpose verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    }
    
    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    
    return 0;
}
