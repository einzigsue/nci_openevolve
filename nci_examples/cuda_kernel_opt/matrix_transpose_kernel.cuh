#ifndef MATRIX_TRANSPOSE_KERNEL_CUH
#define MATRIX_TRANSPOSE_KERNEL_CUH

__global__ void matrixTranspose(float* out, const float* in, int width, int height);

#endif

