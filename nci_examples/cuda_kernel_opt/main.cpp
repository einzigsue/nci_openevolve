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

    // Allocate device arrays
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copy input to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((width + 15)/16, (height + 15)/16);
    matrixTranspose<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    cudaDeviceSynchronize();

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
    delete[] h_in;
    delete[] h_out;
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
