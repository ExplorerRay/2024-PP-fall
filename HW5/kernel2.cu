#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define THREAD_SIZE 32

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

__global__ void mandelKernel(float x1, float y1, float x0, float y0, int* output, int width, int height, int maxIterations, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        float dx = (x1 - x0) / width;
        float dy = (y1 - y0) / height;

        float x = x0 + i * dx;
        float y = y0 + j * dy;

        int *row = (int*)((char*)output + j * pitch);
        row[i] = mandel(x, y, maxIterations);
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // init for host memory
    int *H_mem;
    cudaHostAlloc((void**)&H_mem, resX * resY * sizeof(int), cudaHostAllocDefault);

    // init for device(GPU) memory
    int *D_mem;
    size_t pitch;
    cudaMallocPitch((void**)&D_mem, &pitch, resX * sizeof(int), resY);

    int newResX = resX + (THREAD_SIZE - resX % THREAD_SIZE);
    int newResY = resY + (THREAD_SIZE - resY % THREAD_SIZE);
    dim3 numBlocks(newResX/THREAD_SIZE, newResY/THREAD_SIZE);
    dim3 threadsPerBlock(THREAD_SIZE, THREAD_SIZE);
    // make resX and resY be multiple of THREAD_SIZE
    mandelKernel<<<numBlocks, threadsPerBlock>>>(upperX, upperY, lowerX, lowerY, D_mem, resX, resY, maxIterations, pitch);

    cudaMemcpy2D(H_mem, resX * sizeof(int), D_mem, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, H_mem, resX * resY * sizeof(int));

    cudaFreeHost(H_mem);
    cudaFree(D_mem);
}
