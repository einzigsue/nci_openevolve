#include "matrix_transpose_kernel.cuh"

__global__ void matrixTranspose(float* out, const float* in, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int in_idx  = y * width + x;
        int out_idx = x * height + y;
        out[out_idx] = in[in_idx];
    }
}
