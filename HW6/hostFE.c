#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;

    // create buffers for image, filter and output
    cl_mem imageBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, imageSize * sizeof(float), inputImage, &status);
    cl_mem filterBuffer = clCreateBuffer(*context, CL_MEM_USE_HOST_PTR, filterSize * sizeof(float), filter, &status);
    cl_mem outputBuffer = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, &status);

    // create kernel
    cl_kernel kernel = clCreateKernel(*program, "convolution", &status);

    // set args
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &filterBuffer);
    status = clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    status = clSetKernelArg(kernel, 4, sizeof(int), &imageWidth);
    status = clSetKernelArg(kernel, 5, sizeof(int), &filterWidth);

    // create command queue
    cl_command_queue queue = clCreateCommandQueue(*context, *device, 0, &status);

    // enqueue kernel
    const size_t globalSize[2] = {imageWidth, imageHeight};
    const size_t localSize[2] = {20, 20};
    status = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // read output buffer
    status = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0,imageSize * sizeof(float), outputImage, 0, NULL, NULL);
}
