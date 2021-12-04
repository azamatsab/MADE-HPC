#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__
void make_hist(uint8_t* inputChannel,
              int* hist,
              int numRows, int numCols)
{
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= numCols || py >= numRows) {
      return;
  }
 
  if (inputChannel[py * numCols + px] > 255)
    printf("Error\n");
  
  atomicAdd((int *)(hist + inputChannel[py * numCols + px]), 1);  
}

int main() {
    int width, height, bpp;

    uint8_t* image = stbi_load("gray_image.jpg", &width, &height, &bpp, 1);
    int* hist = (int *) malloc(256 * sizeof(int));

    uint8_t* devImage;
    int* devHist;

    cudaMalloc(&devImage, width * height * sizeof(uint8_t));
    cudaMalloc(&devHist, 256 * sizeof(int));

    cudaMemcpy(devImage, image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    make_hist<<<dim3(100, 100, 1), dim3(3, 3, 1)>>>(devImage, devHist, height, width);
    cudaDeviceSynchronize();
    cudaMemcpy(hist, devHist, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    FILE *f = fopen("histogram.csv", "w"); 
    if (f == NULL) return -1;
    for (int i = 0; i < 256; i++) {
      fprintf(f, "%d,", hist[i]); 
    }
    fclose(f);

    free(hist);
    cudaFree(devImage);
    cudaFree(devHist);
    stbi_image_free(image);

    return 0;
}