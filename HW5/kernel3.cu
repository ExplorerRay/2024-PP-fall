#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <numeric>

#define THREAD_SIZE 16
#define GROUP_SIZE 2

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

    int startX = i * GROUP_SIZE;
    int startY = j * GROUP_SIZE;

    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    for (int y = startY; y < startY + GROUP_SIZE && y < height; ++y) {
        for (int x = startX; x < startX + GROUP_SIZE && x < width; ++x) {
            float md_x = x0 + x * dx;
            float md_y = y0 + y * dy;

            int *row = (int*)((char*)output + y * pitch);
            row[x] = mandel(md_x, md_y, maxIterations);
        }
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

    // make resX and resY be multiple of THREAD_SIZE and GROUP_SIZE
    int T_G_LCM = std::lcm(THREAD_SIZE, GROUP_SIZE);
    int newResX = resX + (T_G_LCM - resX % T_G_LCM);
    int newResY = resY + (T_G_LCM - resY % T_G_LCM);
    dim3 numBlocks(newResX/T_G_LCM, newResY/T_G_LCM);
    dim3 threadsPerBlock(THREAD_SIZE, THREAD_SIZE);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(upperX, upperY, lowerX, lowerY, D_mem, resX, resY, maxIterations, pitch);

    cudaMemcpy2D(H_mem, resX * sizeof(int), D_mem, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, H_mem, resX * resY * sizeof(int));

    cudaFreeHost(H_mem);
    cudaFree(D_mem);
}
