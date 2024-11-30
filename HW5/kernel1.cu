#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define THREAD_SIZE 16

__global__ void mandelKernel(float stepX, float stepY, float x0, float y0, int* output, int width, int height, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height) {
        float c_re = x0 + i * stepX;
        float c_im = y0 + j * stepY;
        float z_re = c_re, z_im = c_im;

        int k;
        for (k = 0; k < maxIterations; ++k) {
            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
        output[j * width + i] = k;
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    int mem_size = resX * resY * sizeof(int);

    // init for host memory
    int *H_mem = (int*)malloc(mem_size);

    // init for device(GPU) memory
    int *D_mem;
    cudaMalloc((void**)&D_mem, mem_size);

    int newResX = resX + (THREAD_SIZE - resX % THREAD_SIZE);
    int newResY = resY + (THREAD_SIZE - resY % THREAD_SIZE);
    dim3 numBlocks(newResX/THREAD_SIZE, newResY/THREAD_SIZE);
    dim3 threadsPerBlock(THREAD_SIZE, THREAD_SIZE);
    // make resX and resY be multiple of THREAD_SIZE
    mandelKernel<<<numBlocks, threadsPerBlock>>>(stepX, stepY, lowerX, lowerY, D_mem, resX, resY, maxIterations);

    cudaMemcpy(H_mem, D_mem, mem_size, cudaMemcpyDeviceToHost);
    memcpy(img, H_mem, mem_size);

    free(H_mem);
    cudaFree(D_mem);
}
