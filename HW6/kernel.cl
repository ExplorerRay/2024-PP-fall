__kernel void convolution(
    const __global float* inputImage,
    __global float* outputImage,
    const __global float* filter,
    const int imageHeight,
    const int imageWidth,
    const int filterWidth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int halfFilterSize = filterWidth / 2;
    int k, l;
    float sum = 0.0f;

    // iterate over the filter
    for (k = -halfFilterSize;k <= halfFilterSize;k++) {
        for (l = -halfFilterSize; l <= halfFilterSize; l++)
        {
            if(filter[(k + halfFilterSize) * filterWidth + l + halfFilterSize] != 0)
            {
                if (y + k >= 0 && y + k < imageHeight &&
                    x + l >= 0 && x + l < imageWidth)
                {
                    sum += inputImage[(y + k) * imageWidth + x + l] *
                            filter[(k + halfFilterSize) * filterWidth +
                                    l + halfFilterSize];
                }
            }
        }
    }

    outputImage[y * imageWidth + x] = sum;
}
