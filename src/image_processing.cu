#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// ---------------- GRAYSCALE KERNEL ----------------

__global__ void grayscaleKernel(
    const unsigned char* input,
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
            0.299f * input[idx + 2] +
            0.587f * input[idx + 1] +
            0.114f * input[idx];

        output[y * width + x] = gray;
    }
}

// ---------------- GAUSSIAN BLUR KERNEL ----------------

__global__ void gaussianKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int kernel[5][5] = {
            {1,  4,  6,  4, 1},
            {4, 16, 24, 16, 4},
            {6, 24, 36, 24, 6},
            {4, 16, 24, 16, 4},
            {1,  4,  6,  4, 1}
        };

        float sum = 0.0f;

        for (int ky = -2; ky <= 2; ky++)
        {
            for (int kx = -2; kx <= 2; kx++)
            {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                sum += input[py * width + px] *
                       kernel[ky + 2][kx + 2];
            }
        }

        output[y * width + x] = (unsigned char)(sum / 256.0f);
    }
}

// ---------------- SOBEL EDGE DETECTION KERNEL ----------------

__global__ void sobelKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int gx = 0;
        int gy = 0;

        int sobelX[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        int sobelY[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        for (int ky = -1; ky <= 1; ky++)
        {
            for (int kx = -1; kx <= 1; kx++)
            {
                int px = min(max(x + kx, 0), width - 1);
                int py = min(max(y + ky, 0), height - 1);

                int pixel = input[py * width + px];

                gx += pixel * sobelX[ky + 1][kx + 1];
                gy += pixel * sobelY[ky + 1][kx + 1];
            }
        }

        float magnitude = sqrtf((float)(gx * gx + gy * gy));

        output[y * width + x] =
            (unsigned char)min(magnitude, 255.0f);
    }
}

// ---------------- GPU PROCESSING ----------------

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

    size_t inputSize = input.total() * input.elemSize();
    size_t imageSize = width * height;

    grayscale.create(height, width, CV_8UC1);
    blurred.create(height, width, CV_8UC1);
    edges.create(height, width, CV_8UC1);

    unsigned char *d_input, *d_gray, *d_blur, *d_edges;

    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_gray, imageSize);
    cudaMalloc((void**)&d_blur, imageSize);
    cudaMalloc((void**)&d_edges, imageSize);

    cudaMemcpy(
        d_input, input.data, inputSize,
        cudaMemcpyHostToDevice
    );

    dim3 block(16, 16);

    dim3 grid(
        (width + 15) / 16,
        (height + 15) / 16
    );

    // 1. GPU Grayscale
    grayscaleKernel<<<grid, block>>>(
        d_input, d_gray, width, height, channels
    );

    // 2. GPU Gaussian Blur
    gaussianKernel<<<grid, block>>>(
        d_gray, d_blur, width, height
    );

    // 3. GPU Sobel Edge Detection
    sobelKernel<<<grid, block>>>(
        d_blur, d_edges, width, height
    );

    cudaDeviceSynchronize();

    // Copy GPU results back to CPU
    cudaMemcpy(
        grayscale.data, d_gray, imageSize,
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        blurred.data, d_blur, imageSize,
        cudaMemcpyDeviceToHost
    );

    cudaMemcpy(
        edges.data, d_edges, imageSize,
        cudaMemcpyDeviceToHost
    );

    cudaFree(d_input);
    cudaFree(d_gray);
    cudaFree(d_blur);
    cudaFree(d_edges);
}

// ---------------- CPU PROCESSING ----------------

void processImageCPU(
    const cv::Mat& input,
    cv::Mat& grayscale,
    cv::Mat& blurred,
    cv::Mat& edges
)
{
    cv::cvtColor(
        input, grayscale, cv::COLOR_BGR2GRAY
    );

    cv::GaussianBlur(
        grayscale, blurred,
        cv::Size(5, 5), 1.5
    );

    cv::Mat gradX, gradY, magnitude;

    cv::Sobel(
        blurred, gradX, CV_16S, 1, 0
    );

    cv::Sobel(
        blurred, gradY, CV_16S, 0, 1
    );

    cv::convertScaleAbs(gradX, gradX);
    cv::convertScaleAbs(gradY, gradY);

    cv::addWeighted(
        gradX, 0.5, gradY, 0.5, 0, edges
    );
}
