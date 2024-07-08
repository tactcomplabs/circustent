#include <cuda_runtime.h>
#include <stdio.h>

void printCudaDeviceInfo() {
  int deviceCount;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
    return;
  }

  if (deviceCount == 0) {
    printf("There are no available CUDA devices.\n");
  } else {
    for (int dev = 0; dev < deviceCount; ++dev) {
      struct cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, dev);
      printf("Device %d: \"%s\"\n", dev, deviceProp.name);
      printf("  Total amount of global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
      printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
      printf("  Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
      printf("  Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
      printf("  Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
      printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
      printf("  Warp size: %d\n", deviceProp.warpSize);
      printf("  Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
      printf("  Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf("  Maximum sizes of each dimension of a block: %d x %d x %d\n",
             deviceProp.maxThreadsDim[0],
             deviceProp.maxThreadsDim[1],
             deviceProp.maxThreadsDim[2]);
      printf("  Maximum sizes of each dimension of a grid: %d x %d x %d\n",
             deviceProp.maxGridSize[0],
             deviceProp.maxGridSize[1],
             deviceProp.maxGridSize[2]);
      printf("  Clock rate: %.2f GHz\n", deviceProp.clockRate / 1.0e6);
      printf("  Memory clock rate: %.2f GHz\n", deviceProp.memoryClockRate / 1.0e6);
      printf("  Memory bus width: %d bits\n", deviceProp.memoryBusWidth);
      printf("  L2 cache size: %d bytes\n", deviceProp.l2CacheSize);
      printf("  Maximum texture dimensions: 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
             deviceProp.maxTexture1D,
             deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
             deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
      printf("\n");
    }
  }
}

int main() {
  printCudaDeviceInfo();
  return 0;
}

