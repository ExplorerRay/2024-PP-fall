#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define THREAD_SIZE 16

__device__ int mandel(float2 c, int count) {
    float2 z = c;
    int i;
    for (i = 0; i < count; ++i)
    {
        if (z.x * z.x + z.y * z.y > 4.f)
            break;

        float2 new_iter = make_float2(z.x * z.x - z.y * z.y, 2.f * z.x * z.y);
        z.x = c.x + new_iter.x;
        z.y = c.y + new_iter.y;
    }
    return i;
}

__global__ void mandelKernel(float stepX, float stepY, float x0, float y0, int* output, int width, int height, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int2 pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

    if (pos.x < width && pos.y < height) {
        float2 posf = make_float2(x0 + pos.x * stepX, y0 + pos.y * stepY);

        int index = (pos.y * width + pos.x);
        output[index] = mandel(posf, maxIterations);
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // init for device(GPU) memory
    int *D_mem;
    cudaMalloc((void**)&D_mem, resX * resY * sizeof(int));

    int newResX = resX + (THREAD_SIZE - resX % THREAD_SIZE);
    int newResY = resY + (THREAD_SIZE - resY % THREAD_SIZE);
    dim3 numBlocks(newResX/THREAD_SIZE, newResY/THREAD_SIZE);
    dim3 threadsPerBlock(THREAD_SIZE, THREAD_SIZE);
    // make resX and resY be multiple of THREAD_SIZE
    mandelKernel<<<numBlocks, threadsPerBlock>>>(stepX, stepY, lowerX, lowerY, D_mem, resX, resY, maxIterations);

    cudaMemcpy(img, D_mem, resX * resY * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(D_mem);
}
