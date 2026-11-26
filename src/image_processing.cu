#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

__global__ void grayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * channels;

        unsigned char gray =
            0.299f * input[idx] +
            0.587f * input[idx + 1] +
            0.114f * input[idx + 2];

        output[y * width + x] = gray;
    }
}

void processImageGPU(
    const cv::Mat& input,
    cv::Mat& grayscale,
    cv::Mat& blurred,
    cv::Mat& edges
)
{
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();

    grayscale.create(height, width, CV_8UC1);

    unsigned char *d_input, *d_gray;

    size_t inputSize = input.total() * input.elemSize();
    size_t graySize = width * height;

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_gray, graySize);

    cudaMemcpy(
        d_input,
        input.data,
        inputSize,
        cudaMemcpyHostToDevice
    );

    dim3 block(16, 16);
    dim3 grid(
        (width + 15) / 16,
        (height + 15) / 16
    );

    grayscaleKernel<<<grid, block>>>(
        d_input,
        d_gray,
        width,
        height,
        channels
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        grayscale.data,
        d_gray,
        graySize,
        cudaMemcpyDeviceToHost
    );

    cv::GaussianBlur(
        grayscale,
        blurred,
        cv::Size(5, 5),
        1.5
    );

    cv::Sobel(
        blurred,
        edges,
        CV_8U,
        1,
        1
    );

    cudaFree(d_input);
    cudaFree(d_gray);
}
