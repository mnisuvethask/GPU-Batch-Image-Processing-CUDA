## GPU-Accelerated Batch Image Processing Using CUDA

## Project Overview

GPU-Accelerated Batch Image Processing is a parallel computing project that uses NVIDIA CUDA to process multiple high-resolution images efficiently. The project implements image-processing operations such as grayscale conversion, Gaussian blur, and Sobel edge detection using GPU parallel processing.

The system processes a batch of 10 high-resolution images (1080 × 1980 pixels) and compares CPU-based processing with GPU-based processing to analyse execution time, computational efficiency, and performance improvement.

By utilizing CUDA threads to execute image-processing operations in parallel, the project demonstrates how GPU computing can accelerate computationally intensive tasks compared to traditional CPU processing.

---

## Objectives

- To implement image-processing operations using CUDA.
- To understand GPU parallel processing and thread execution.
- To process multiple high-resolution images in batches.
- To compare CPU and GPU execution time.
- To analyse GPU performance using speedup calculations.

---

## Image Processing Operations

1. Grayscale Conversion

Converts colour images into grayscale images by transforming pixel colour values into intensity values.

2. Gaussian Blur

Applies Gaussian filtering to reduce noise and smooth the image by averaging neighbouring pixel values using a Gaussian kernel.

3. Sobel Edge Detection

Detects edges in images by calculating intensity gradients in horizontal and vertical directions to identify boundaries and structural details.

---

## CUDA Implementation – Main Code

1. Grayscale Conversion Kernel

CUDA threads process individual pixels in parallel. Each thread calculates the grayscale intensity of one pixel using its Blue, Green, and Red channel values.

__global__ void grayscaleKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * 3;

        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];

        output[y * width + x] = (r + g + b) / 3;
    }
}

2. Gaussian Blur Kernel

Gaussian blur smooths the image by applying a 5 × 5 filter to each pixel using neighbouring pixel values.

__global__ void gaussianBlurKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int filter[5][5] = {
        { 1,  4,  6,  4,  1},
        { 4, 16, 24, 16,  4},
        { 6, 24, 36, 24,  6},
        { 4, 16, 24, 16,  4},
        { 1,  4,  6,  4,  1}
    };

    int sum = 0;

    for (int i = -2; i <= 2; i++)
    {
        for (int j = -2; j <= 2; j++)
        {
            int nx = x + j;
            int ny = y + i;

            if (nx >= 0 && nx < width &&
                ny >= 0 && ny < height)
            {
                sum += input[ny * width + nx]
                       * filter[i + 2][j + 2];
            }
        }
    }

    output[y * width + x] = sum / 256;
}

3. Sobel Edge Detection Kernel

Sobel edge detection calculates horizontal and vertical intensity gradients to identify edges in an image.

__global__ void sobelKernel(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

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

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            int nx = x + j;
            int ny = y + i;

            if (nx >= 0 && nx < width &&
                ny >= 0 && ny < height)
            {
                int pixel = input[ny * width + nx];

                gx += pixel * sobelX[i + 1][j + 1];
                gy += pixel * sobelY[i + 1][j + 1];
            }
        }
    }

    float magnitude = sqrtf(
        (float)(gx * gx + gy * gy));

    output[y * width + x] =
        (unsigned char)(magnitude > 255 ? 255 : magnitude);
}

4. CUDA Execution Configuration

The image is divided into blocks of threads. Each thread is assigned a pixel to process, allowing thousands of pixels to be processed concurrently.

dim3 blockSize(16, 16);

dim3 gridSize(
    (width + blockSize.x - 1) / blockSize.x,
    (height + blockSize.y - 1) / blockSize.y
);

grayscaleKernel<<<gridSize, blockSize>>>(
    d_input, d_gray, width, height
);

gaussianBlurKernel<<<gridSize, blockSize>>>(
    d_gray, d_blur, width, height
);

sobelKernel<<<gridSize, blockSize>>>(
    d_gray, d_edges, width, height
);

cudaDeviceSynchronize();

---

## Technologies Used

Technology| Purpose
C++| Application development
NVIDIA CUDA| GPU parallel processing
OpenCV| Image loading and CPU image processing
CMake| Build configuration
Google Colab| Development and execution environment
NVIDIA Tesla T4| GPU acceleration

---

## Dataset

- Number of images: 10
- Image resolution: 1080 × 1980 pixels
- Image format: JPG
- Processing mode: Batch processing

---

## Project Structure

GPU-Batch-Image-Processing/
│
├── README.md
├── CMakeLists.txt
│
├── src/
│   ├── batch_processor.cpp
│   ├── image_processing.cu
│   └── image_processing.h
│
├── dataset/
│   └── sample_images/
│
├── output/
│   └── processed_images.zip
│
├── results/
│   └── performance.csv
│
└── screenshots/

---

## Workflow

1. Load input images from the dataset directory.
2. Perform image processing using CPU implementation.
3. Perform image processing using CUDA GPU implementation.
4. Generate grayscale, Gaussian blur, and edge-detection outputs.
5. Measure CPU and GPU execution times.
6. Calculate performance speedup.
7. Store processed images and performance results.

---

## Performance Analysis

The project compares CPU and GPU execution time for batch image processing.

Metric| Description
CPU Execution Time| Time taken by CPU to process the batch
GPU Execution Time| Time taken by GPU to process the batch
Speedup| CPU execution time divided by GPU execution time

Speedup Formula

[
\text{Speedup} = \frac{\text{CPU Execution Time}}{\text{GPU Execution Time}}
]

Performance results are stored in "results/performance.csv".

---

## Expected Outcomes

- Faster image-processing execution using GPU parallelism.
- Understanding of CUDA kernel execution.
- Comparison of CPU and GPU computational performance.
- Demonstration of parallel computing for image-processing applications.

---

## Conclusion

This project demonstrates the application of NVIDIA CUDA in accelerating batch image-processing tasks using GPU parallel computing. By implementing grayscale conversion, Gaussian blur, and Sobel edge detection, the project evaluates the performance differences between CPU and GPU implementations and highlights the benefits of parallel processing for computationally intensive workloads.

---

Developed as a Parallel Computing / GPU Computing project.
