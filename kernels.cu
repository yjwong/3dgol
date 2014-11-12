#include <iostream>
#include <cstdio>
#include "util.h"
#include "kernels.h"

#define GOL_CUDA_BLOCK_X 4
#define GOL_CUDA_BLOCK_Y 4
#define GOL_CUDA_BLOCK_Z 4
#define GOL_CUDA_BLOCK_COUNT 64

namespace GameOfLife {

__device__ int golKernelGetOffset(int x, int y, int z, int size) {
    return x * size * size + y * size + z;
}

__device__ int golKernelGetSharedOffset(int x, int y, int z) {
    return x * GOL_CUDA_BLOCK_X * GOL_CUDA_BLOCK_X + y * GOL_CUDA_BLOCK_Y + z;
}

__device__ int golKernelHasNeighbour(int x, int y, int z, int i, int j, int k,
        int size, int* data, int* dataS) {
    int testSharedX = threadIdx.x + i;
    int testSharedY = threadIdx.y + j;
    int testSharedZ = threadIdx.z + k;

    int testX = x + i;
    int testY = y + j;
    int testZ = z + k;

    // Implement wraparound for the cube.
    if (testX < 0) {
        testX = size - 1;
    }

    if (testX >= size) {
        testX = 0;
    }

    if (testY < 0) {
        testY = size - 1;
    }

    if (testY >= size) {
        testY = 0;
    }

    if (testZ < 0) {
        testZ = size - 1;
    }

    if (testZ >= size) {
        testZ = 0;
    }

    // Check if shared memory has requested contents.
    bool useSharedMemory = !(
        testX < 0 || testX >= size ||
        testY < 0 || testY >= size ||
        testZ < 0 || testZ >= size ||
        testSharedX < 0 || testSharedX >= GOL_CUDA_BLOCK_X ||
        testSharedY < 0 || testSharedY >= GOL_CUDA_BLOCK_Y ||
        testSharedZ < 0 || testSharedZ >= GOL_CUDA_BLOCK_Z);

    // Ignore self.
    // Access the value and count.
    if (i != 0 || j != 0 || k != 0) {
        if (useSharedMemory) {
            return dataS[golKernelGetSharedOffset(testSharedX,
                  testSharedY, testSharedZ)]; 
        } else {
            return data[golKernelGetOffset(testX, testY, testZ, size)];
        }
    }

    return 0;
}

__global__ void golKernelSharedMemory(int* data, int* outData, int size,
        int r1, int r2, int r3, int r4) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Deal with non-power-of-two sizes.
    // Some threads would be left idle as a consequence.
    if (x >= size || y >= size || z >= size) {
        return;
    }

    // Get offset of the shared data.
    int dataOffset = golKernelGetOffset(x, y, z, size); 
    int sharedDataOffset = golKernelGetSharedOffset(threadIdx.x, threadIdx.y,
            threadIdx.z);

    // Use shared memory.
    int threadLinearId = threadIdx.x * GOL_CUDA_BLOCK_X * GOL_CUDA_BLOCK_X +
        threadIdx.y * GOL_CUDA_BLOCK_Y + threadIdx.z;
    __shared__ int dataS[GOL_CUDA_BLOCK_COUNT];
    dataS[threadLinearId] = data[dataOffset];
    __syncthreads();

    // Check neighbours.
    int sum = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {
                sum += golKernelHasNeighbour(x, y, z, i, j, k, size, data,
                        dataS);
            }
        }
    }

    // Apply the rules.
    int thisCell = dataS[sharedDataOffset];
    if (thisCell == 0) {
        if (sum >= r1 && sum <= r2) {
            outData[dataOffset] = 1;
        } else {
            outData[dataOffset] = 0;
        }
    } else {
        if (sum > r3 || sum < r4) {
            outData[dataOffset] = 0;
        } else {
            outData[dataOffset] = 1;
        }
    }

}

__global__ void golKernel(int* data, int* outData, int size, int r1, int r2,
        int r3, int r4) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    // Deal with non-power-of-two sizes.
    // Some threads would be left idle as a consequence.
    if (x >= size || y >= size || z >= size) {
        return;
    }

    int sum = 0;
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            for (int k = -1; k < 2; k++) {
                int testX = x + i;
                int testY = y + j;
                int testZ = z + k;

                // Implement wraparound for the cube.
                if (testX < 0) {
                    testX = size - 1;
                }

                if (testX >= size) {
                    testX = 0;
                }

                if (testY < 0) {
                    testY = size - 1;
                }

                if (testY >= size) {
                    testY = 0;
                }

                if (testZ < 0) {
                    testZ = size - 1;
                }

                if (testZ >= size) {
                    testZ = 0;
                }

                // Ignore self.
                // Access the value and count.
                if (i != 0 || j != 0 || k != 0) {
                    sum += data[golKernelGetOffset(testX, testY, testZ, size)];
                }
            }
        }
    }

    // Apply the rules.
    int dataOffset = golKernelGetOffset(x, y, z, size);
    int thisCell = data[dataOffset];
    if (thisCell == 0) {
        if (sum >= r1 && sum <= r2) {
            outData[dataOffset] = 1;
        } else {
            outData[dataOffset] = 0;
        }
    } else {
        if (sum > r3 || sum < r4) {
            outData[dataOffset] = 0;
        } else {
            outData[dataOffset] = 1;
        }
    }
}

__host__ void golPerformSimulation(Field* field, Field* outField, int r1,
        int r2, int r3, int r4) {
    dim3 grid = dim3((field->size() + GOL_CUDA_BLOCK_X - 1) / GOL_CUDA_BLOCK_X,
            (field->size() + GOL_CUDA_BLOCK_Y - 1) / GOL_CUDA_BLOCK_Y,
            (field->size() + GOL_CUDA_BLOCK_Z - 1) / GOL_CUDA_BLOCK_Z);
    dim3 block = dim3(GOL_CUDA_BLOCK_X, GOL_CUDA_BLOCK_Y, GOL_CUDA_BLOCK_Z);

    // Prepare the buffers.
    int* data = field->allocateOnDevice();
    field->transferToDevice();
    int* outData = outField->allocateOnDevice();

    // Run the kernel.
    golKernelSharedMemory<<<grid, block>>> (data, outData, field->size(), r1,
            r2, r3, r4);
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        Util::cudaThrowError("Unable to run kernel", result);
    }

    result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        Util::cudaThrowError("Unable to synchronize device", result);
    }

    outField->transferFromDevice();
    cudaDeviceSynchronize();

    // Free memory.
    cudaFree(data);
    cudaFree(outData);
}

__host__ void golPerformSimulationMulti(Field* field, Field* outField, int r1,
        int r2, int r3, int r4, int count) {
    dim3 grid = dim3(field->size() / GOL_CUDA_BLOCK_X,
            field->size() / GOL_CUDA_BLOCK_Y,
            field->size() / GOL_CUDA_BLOCK_Z);
    dim3 block = dim3(GOL_CUDA_BLOCK_X, GOL_CUDA_BLOCK_Y, GOL_CUDA_BLOCK_Z);

    // Measure time taken.
    float totalTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Prepare the buffers.
    int* data = field->allocateOnDevice();
    field->transferToDevice();
    int* outData = outField->allocateOnDevice();

    // Run the kernel multiple times.
    for (int i = 0; i < count; i++) {
        cudaEventRecord(start);
        golKernelSharedMemory<<<grid, block>>> (data, outData, field->size(),
                r1, r2, r3, r4);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float iterationTime = 0;
        cudaEventElapsedTime(&iterationTime, start, stop);
        totalTime += iterationTime;

        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess) {
            Util::cudaThrowError("Unable to run kernel", result);
        }

        result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            Util::cudaThrowError("Unable to synchronize device", result);
        }

        // Swap input and output.
        int* temp = outData;
        outData = data;
        data = temp;
    }

    outField->transferFromDevice();
    cudaDeviceSynchronize();

    std::cout << "Kernel execution time: " << totalTime << " ms" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free memory.
    cudaFree(data);
    cudaFree(outData);
}

}

/* vim: set ts=4 sw=4 et: */
