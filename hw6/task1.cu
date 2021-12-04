#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__
void convolve(uint8_t* inputChannel,
              uint8_t* outputChannel,
              int numRows, int numCols,
              float* filter, size_t filterWidth, float sum)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= numCols || py >= numRows) {
      return;
  }

  float c = 0.0f;

  for (int fx = 0; fx < filterWidth; fx++) {
    for (int fy = 0; fy < filterWidth; fy++) {
      int imagex = px + fx - filterWidth / 2;
      int imagey = py + fy - filterWidth / 2;
      imagex = min(max(imagex, 0), numCols - 1);
      imagey = min(max(imagey, 0), numRows - 1);
      c += (filter[fy * filterWidth + fx] * inputChannel[imagey * numCols + imagex]);
    }
  }

  outputChannel[py * numCols + px] = c / sum;
}

int main() {
    int width, height, bpp;
    int filterSize = 3;
    float* filter = (float *)malloc(filterSize * filterSize * sizeof(float));

    double sum = 0;
    for (int i = 0; i < filterSize * filterSize; i++) {
        filter[i] = 0.1;
    }
    for (int i = 0; i < filterSize * filterSize; i++) {
        sum += filter[i];
    }

    uint8_t* image = stbi_load("gray_image.jpg", &width, &height, &bpp, 1);
    uint8_t* outImage = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    uint8_t* devImage;
    uint8_t* devOutImage;
    float* devFilter;

    cudaMalloc(&devImage, width * height * sizeof(uint8_t));
    cudaMalloc(&devOutImage, width * height * sizeof(uint8_t));
    cudaMalloc(&devFilter, filterSize * filterSize * sizeof(float));

    cudaMemcpy(devImage, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(devOutImage, outImage, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(devFilter, filter, filterSize * filterSize * sizeof(float), cudaMemcpyHostToDevice);

    convolve<<<dim3(100, 100, 1), dim3(3, 3, 1)>>>(devImage, devOutImage, height, width, devFilter, filterSize, sum);
    cudaDeviceSynchronize();
    cudaMemcpy(outImage, devOutImage, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    stbi_write_png("task1_3x3.png", width, height, 1, outImage, width * 1);

    free(outImage);
    free(filter);
    cudaFree(devFilter);
    cudaFree(devImage);
    cudaFree(devOutImage);
    stbi_image_free(image);

    return 0;
}