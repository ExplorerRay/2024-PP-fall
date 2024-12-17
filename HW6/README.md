# PP HW6 report (110550123)

## Q1
> Explain your implementation. How do you optimize the performance of convolution?

About `hostFE.c`, just do some preparations (buffer, command queue, kernel) for OpenCL. And call `clEnqueueNDRangeKernel` to run the kernel. Then read buffer to target output.

About `kernel.cl`, I just rewrite `serialConv.c` to OpenCL kernel.

I use 2D work group to do convolution because I think 2D will be more easily to understand and implement instead of 1D.

About performance, in `kernel.cl`, I check for filter content is 0 or not, if it is 0, I will not do convolution.
In `hostFE.c`, I try to enlarge `localSize` to make more threads run.

I am not familiar how to optimize the performance of convolution, so I just do some simple optimization.

## Q2
> Rewrite the program using CUDA. (1) Explain your CUDA implementation, (2) plot a chart to show the performance difference between using OpenCL and CUDA, and (3) explain the result.
