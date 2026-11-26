#include "image_processing.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace chrono;
namespace fs = filesystem;

int main()
{
    string inputFolder = "dataset/sample_images/";
    string outputFolder = "output/";

    fs::create_directories(outputFolder);

    double totalCPU = 0;
    double totalGPU = 0;

    int imageCount = 0;

    for (const auto& entry : fs::directory_iterator(inputFolder))
    {
        if (!entry.is_regular_file())
            continue;

        string filePath = entry.path().string();

        Mat input = imread(filePath);

        if (input.empty())
        {
            cout << "Skipping: " << filePath << endl;
            continue;
        }

        Mat grayCPU, blurCPU, edgeCPU;
        Mat grayGPU, blurGPU, edgeGPU;

        // CPU processing
        auto cpuStart = high_resolution_clock::now();

        processImageCPU(
            input,
            grayCPU,
            blurCPU,
            edgeCPU
        );

        auto cpuEnd = high_resolution_clock::now();

        double cpuTime =
            duration<double, milli>(cpuEnd - cpuStart).count();

        // GPU processing
        auto gpuStart = high_resolution_clock::now();

        processImageGPU(
            input,
            grayGPU,
            blurGPU,
            edgeGPU
        );

        auto gpuEnd = high_resolution_clock::now();

        double gpuTime =
            duration<double, milli>(gpuEnd - gpuStart).count();

        totalCPU += cpuTime;
        totalGPU += gpuTime;
        imageCount++;

        // Save GPU outputs
        string name = entry.path().stem().string();

        imwrite(outputFolder + name + "_grayscale.jpg", grayGPU);
        imwrite(outputFolder + name + "_blur.jpg", blurGPU);
        imwrite(outputFolder + name + "_edges.jpg", edgeGPU);

        cout << "Processed: " << name << endl;
        cout << "CPU Time: " << cpuTime << " ms" << endl;
        cout << "GPU Time: " << gpuTime << " ms" << endl;
        cout << "-----------------------------" << endl;
    }

    if (imageCount > 0)
    {
        cout << "\n===== PERFORMANCE SUMMARY =====" << endl;
        cout << "Images processed: " << imageCount << endl;
        cout << "Total CPU Time: " << totalCPU << " ms" << endl;
        cout << "Total GPU Time: " << totalGPU << " ms" << endl;
        cout << "Speedup: " << totalCPU / totalGPU << "x" << endl;
    }

    return 0;
}
