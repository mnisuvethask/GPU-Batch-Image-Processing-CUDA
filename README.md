GPU-Accelerated Batch Image Processing Using CUDA

Overview

This project implements batch image processing using GPU acceleration with CUDA. Multiple high-resolution images are processed using grayscale conversion, Gaussian blur, and Sobel edge detection.

The project compares CPU and GPU processing times to study the performance of parallel computing for image processing.

Objectives

- Process multiple high-resolution images in batches.
- Implement GPU-based image processing using CUDA.
- Perform grayscale conversion, Gaussian blur, and Sobel edge detection.
- Compare CPU and GPU execution times.
- Analyze GPU acceleration and parallel processing performance.

Technologies Used

- C++
- CUDA
- OpenCV
- CMake

Image Processing Operations

1. Grayscale Conversion
2. Gaussian Blur
3. Sobel Edge Detection

Dataset

The project uses high-resolution images of size 1080 × 1980 pixels. Multiple images are processed in a single batch.

Project Structure

GPU-Batch-Image-Processing/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── batch_processor.cpp
│   ├── image_processing.cu
│   └── image_processing.h
├── dataset/
│   └── sample_images/
├── output/
├── results/
│   └── performance.csv
└── screenshots/

Execution

The project is compiled using CMake with CUDA and OpenCV support.

The program loads images from the dataset directory, processes them, saves the output images, and reports execution time.

Performance Analysis

CPU and GPU execution times are measured and compared to evaluate the effectiveness of GPU acceleration.

Author

B.Tech Artificial Intelligence and Data Science Student
