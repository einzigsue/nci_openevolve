#include <iostream>
#include <cuda_runtime.h>
#include "matrix_transpose_kernel.cuh"

int main() {
    int width = 4;
    int height = 3;
    int numElements = width * height;
    size_t size = numElements * sizeof(float);

    // Allocate host arrays
    float* h_in  = new float[numElements];
    float* h_out = new float[numElements];

    // Initialize input with sequential data
    for (int i = 0; i < numElements; ++i)
        h_in[i] = static_cast<float>(i);

    // Print input matrix
    std::cout << "Input matrix (" << height << "x" << width << "):\n";
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            std::cout << h_in[row * width + col] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;

    // Allocate device arrays
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    matrixTranspose<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Kernel execution time: " << milliseconds << " ms\n";

    // Calculate effective memory bandwidth (input + output)
    size_t totalBytes = 2 * width * height * sizeof(float);
    float seconds = milliseconds / 1000.0f;
    float bandwidth = totalBytes / seconds / (1 << 30); // GB/s
    std::cout << "Effective memory bandwidth: " << bandwidth << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // Copy result back to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Print result (matrix form)
    std::cout << "Output (transposed) matrix:\n";
    for (int row = 0; row < width; ++row) {
        for (int col = 0; col < height; ++col)
            std::cout << h_out[row * height + col] << " ";
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
