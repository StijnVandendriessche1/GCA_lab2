#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>

#include <chrono>

// ---- reduction GPU
// -------- GPU recution kernel
__global__ void getMaxReduction(int* A, int* max, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = N;
    for (int n = 0; n < (log2f(N)); n++)
    {
        j = j / 2;
        if (i < j)
        {
            if (A[i] < A[i + j])
            {
                A[i] = A[i + j];
            }
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *max = A[0];
    }
}

//-------- GPU reduction main function
int detectMaxGPUReduction(int* A, int N)
{
    //int for maximum
    int max = 0;

    // Allocate the device input vector
    int* gpuA = NULL;
    cudaMalloc((void**)&gpuA, N * sizeof(int));

    // Allocate the device output int (where the max will be stored)
    int* gpuMax = NULL;
    cudaMalloc((void**)&gpuMax, sizeof(int));

    // Copy the host input vector nd output int in host memory to the device input
    cudaMemcpy(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuMax, &max, sizeof(int), cudaMemcpyHostToDevice);

    //execute kernel
    int threadsPerBlock = 1024;
    //int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = 1;
    getMaxReduction << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuMax, N);

    //copy result back to host
    cudaMemcpy(&max, gpuMax, sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(A, gpuA, N * sizeof(int), cudaMemcpyDeviceToHost);

    //free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuMax);

    //max = A[0];
    return max;
}


// ---- GPU atomic
// -------- GPU atomic kernel
__global__ void getMax(int* A, int* max)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    atomicMax(max, A[i]);
}
// -------- GPU atomic main function
int detectMaxGPU(int* A, int N)
{   
    //int for maximum
    int max = 0;

    // Allocate the device input vector
    int* gpuA = NULL;
    cudaMalloc((void**)&gpuA, N * sizeof(int));

    // Allocate the device output int (where the max will be stored)
    int* gpuMax = NULL;
    cudaMalloc((void**)&gpuMax, sizeof(int));

    // Copy the host input vector nd output int in host memory to the device input
    cudaMemcpy(gpuA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuMax, &max, sizeof(int), cudaMemcpyHostToDevice);

    //execute kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    getMax << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuMax);

    //copy result back to host
    cudaMemcpy(&max, gpuMax, sizeof(int), cudaMemcpyDeviceToHost);

    //free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuMax);

    return max;
}

// ---- CPU
// -------- CPU function
int detectMaxCPU(int* A, int size)
{
    //cpu algorithm
    int max = INT_MIN;
    for (int i = 0; i < size; i++)
    {
        if (A[i] > max)
        {
            max = A[i];
        }
    }
    return max;
}

//main function
int main()
{
    const int base = 2;
    for (int i = 0; i < 11; i++)
    {
        //create a random array of integers
        //allocate memory
        int N = (int)base * pow(2, i); //mount of elements
        size_t size = N * sizeof(int); //amount of bytes
        int* A = (int*)malloc(size); //memory alocation

        //fill up the memory with a random array
        for (int i = 0; i < N; ++i) {
            A[i] = rand();
        }

        //three executions types
        //execute on cpu
        printf("maximum of %d numbers CPU: %d\n\n", N, detectMaxCPU(A, N));

        //execute on GPU
        printf("maximum of %d numbers GPU: %d\n\n", N, detectMaxGPU(A, N));

        //execute on GPU using reduction
        printf("maximum of %d numbers GPU: %d\n\n - - - - \n", N, detectMaxGPUReduction(A, N));

        //free up memory
        free(A);
    }
    return 0;
}
