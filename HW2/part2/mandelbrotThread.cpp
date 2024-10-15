#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <immintrin.h>

#include "CycleTimer.h"

#pragma GCC target("avx2")

typedef struct
{
    float x0, x1;
    float y0, y1;
    unsigned int width;
    unsigned int height;
    int maxIterations;
    int *output;
    int threadId;
    int numThreads;
} WorkerArgs;

static inline __m256i mandel_simd(__m256 c_re, __m256 c_im, int count) {
    __m256 z_re = c_re;
    __m256 z_im = c_im;
    __m256i iter = _mm256_setzero_si256();
    __m256 threshold = _mm256_set1_ps(4.0f);

    for (int i = 0; i < count; ++i) {
        // Calculate z_re^2 and z_im^2
        __m256 z_re2 = _mm256_mul_ps(z_re, z_re);
        __m256 z_im2 = _mm256_mul_ps(z_im, z_im);
        __m256 mag2 = _mm256_add_ps(z_re2, z_im2);

        // Check if the magnitude squared is greater than 4
        __m256 mask = _mm256_cmp_ps(mag2, threshold, _CMP_LE_OQ);
        int mask_int = _mm256_movemask_ps(mask);
        if (mask_int == 0) {
            // All values have escaped, exit early
            break;
        }

        // Calculate the new z_re and z_im
        __m256 new_re = _mm256_sub_ps(z_re2, z_im2);
        __m256 new_im = _mm256_mul_ps(_mm256_mul_ps(z_re, z_im), _mm256_set1_ps(2.0f));
        z_re = _mm256_add_ps(c_re, new_re);
        z_im = _mm256_add_ps(c_im, new_im);

        // Update iterations where mask is true
        __m256i mask_count = _mm256_and_si256(_mm256_castps_si256(mask), _mm256_set1_epi32(1));
        iter = _mm256_add_epi32(iter, mask_count);
    }

    return iter;
}

void mandelbrotCustom(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[],
    int threadID, int numThreads) {
    
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; j += numThreads) {
        float y = y0 + j * dy;
        for (int i = 0; i < width; i += 8) {
            // Calculate x values for 8 pixels at a time
            __m256 x_vals = _mm256_add_ps(_mm256_set1_ps(x0), _mm256_mul_ps(_mm256_set_ps(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i), _mm256_set1_ps(dx)));
            __m256 y_vals = _mm256_set1_ps(y);

            // Perform Mandelbrot iteration for 8 pixels
            __m256i iterations = mandel_simd(x_vals, y_vals, maxIterations);

            // Store the result
            int *iter_array = new int[8];
            _mm256_storeu_si256((__m256i*)iter_array, iterations);
            for (int k = 0; k < 8; ++k) {
                if (i + k < width) {
                    output[(j * width) + (i + k)] = iter_array[k];
                }
            }
        }
    }
}

//
// workerThreadStart --
//
// Thread entrypoint.
void workerThreadStart(WorkerArgs *const args)
{
    double startTime = CycleTimer::currentSeconds();

    mandelbrotCustom(args->x0, args->y0, args->x1, args->y1,
                     args->width, args->height,
                     args->threadId,
                     args->height - args->threadId,
                     args->maxIterations, args->output,
                     args->threadId, args->numThreads);

    double endTime = CycleTimer::currentSeconds();
    printf("Thread %d took %.3f ms\n", args->threadId, (endTime - startTime) * 1000);
}

//
// MandelbrotThread --
//
// Multi-threaded implementation of mandelbrot set image generation.
// Threads of execution are created by spawning std::threads.
void mandelbrotThread(
    int numThreads,
    float x0, float y0, float x1, float y1,
    int width, int height,
    int maxIterations, int output[])
{
    static constexpr int MAX_THREADS = 32;

    if (numThreads > MAX_THREADS)
    {
        fprintf(stderr, "Error: Max allowed threads is %d\n", MAX_THREADS);
        exit(1);
    }

    // Creates thread objects that do not yet represent a thread.
    std::thread workers[MAX_THREADS];
    WorkerArgs args[MAX_THREADS] = {};

    for (int i = 0; i < numThreads; i++)
    {
        args[i].x0 = x0;
        args[i].y0 = y0;
        args[i].x1 = x1;
        args[i].y1 = y1;
        args[i].width = width;
        args[i].height = height;
        args[i].maxIterations = maxIterations;
        args[i].numThreads = numThreads;
        args[i].output = output;

        args[i].threadId = i;
    }

    // Spawn the worker threads.  Note that only numThreads-1 std::threads
    // are created and the main application thread is used as a worker
    // as well.
    for (int i = 1; i < numThreads; i++)
    {
        workers[i] = std::thread(workerThreadStart, &args[i]);
    }

    workerThreadStart(&args[0]); // main application thread is a worker too

    // join worker threads
    for (int i = 1; i < numThreads; i++)
    {
        workers[i].join();
    }
    printf("====================================\n");
}
