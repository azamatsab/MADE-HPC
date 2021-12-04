#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__
void median_filter(uint8_t* inputChannel,
              uint8_t* outputChannel,
              int numRows, int numCols,
              size_t filterWidth)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= numCols || py >= numRows) {
      return;
  }

  uint8_t filter[25];
  
  for (int fx = 0; fx < filterWidth; fx++) {
    for (int fy = 0; fy < filterWidth; fy++) {
      int imagex = px + fx - filterWidth / 2;
      int imagey = py + fy - filterWidth / 2;
      imagex = min(max(imagex, 0), numCols - 1);
      imagey = min(max(imagey, 0), numRows - 1);
      filter[fy * filterWidth + fx] = inputChannel[imagey * numCols + imagex];
    }
  }

  for (int i = 0; i < 25; i++) {
    for (int j = 0; j < 24; j++) {
      if (filter[j] > filter[j + 1]) {
        int b = filter[j];
        filter[j] = filter[j + 1];
        filter[j + 1] = b;
      }
    }
  }
  outputChannel[py * numCols + px] = filter[12];
}

int main() {
    int width, height, bpp;

    uint8_t* image = stbi_load("gray_image.jpg", &width, &height, &bpp, 1);
    uint8_t* outImage = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    uint8_t* devImage;
    uint8_t* devOutImage;

    cudaMalloc(&devImage, width * height * sizeof(uint8_t));
    cudaMalloc(&devOutImage, width * height * sizeof(uint8_t));

    cudaMemcpy(devImage, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(devOutImage, outImage, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    median_filter<<<dim3(100, 100, 1), dim3(3, 3, 1)>>>(devImage, devOutImage, height, width, 5);
    cudaDeviceSynchronize();
    cudaMemcpy(outImage, devOutImage, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    stbi_write_png("task2_median.png", width, height, 1, outImage, width * 1);

    free(outImage);
    cudaFree(devImage);
    cudaFree(devOutImage);
    stbi_image_free(image);

    return 0;
}