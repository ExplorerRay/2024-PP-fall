#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define THREAD_SIZE 16

__device__ int mandel(float c_re, float c_im, int count) {
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i)
    {
        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    return i;
}

__global__ void mandelKernel(float x1, float y1, float x0, float y0, int* output, int width, int height, int maxIterations, int totalRows, int startRow) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < totalRows) {
        float dx = (x1 - x0) / width;
        float dy = (y1 - y0) / height;

        float x = x0 + i * dx;
        float y = y0 + (startRow + j) * dy;

        int index = ((startRow + j) * width + i);
        output[index] = mandel(x, y, maxIterations);
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // init for host memory
    int *H_mem = (int*)malloc(resX * resY * sizeof(int));

    // init for device(GPU) memory
    int *D_mem;
    cudaMalloc((void**)&D_mem, resX * resY * sizeof(int));

    int newResX = resX + (THREAD_SIZE - resX % THREAD_SIZE);
    int newResY = resY + (THREAD_SIZE - resY % THREAD_SIZE);
    dim3 numBlocks(newResX/THREAD_SIZE, newResY/THREAD_SIZE);
    dim3 threadsPerBlock(THREAD_SIZE, THREAD_SIZE);
    // make resX and resY be multiple of THREAD_SIZE
    mandelKernel<<<numBlocks, threadsPerBlock>>>(upperX, upperY, lowerX, lowerY, D_mem, resX, resY, maxIterations, resY, 0);

    cudaMemcpy(H_mem, D_mem, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);
    memcpy(img, H_mem, resX * resY * sizeof(int));

    free(H_mem);
    cudaFree(D_mem);
}
