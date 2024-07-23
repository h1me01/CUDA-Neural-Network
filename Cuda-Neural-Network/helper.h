#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// helper to get the matrix index
#define MI(row, col, numCols) (row * numCols + col)

// for cuda error handling 
static void HandleError(cudaError_t err,
	const char* file,
	int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void checkKernelErrors(const char* kernelName) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("CUDA Error after launching %s: %s\n", kernelName, cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error after launching %s: %s\n", kernelName, cudaGetErrorString(cudaStatus));
	}
}


void allocDevMem(float** d_a, size_t n) {
	HANDLE_ERROR(cudaMalloc((void**)d_a, n * sizeof(float)));
}

void freeDevMem(float* d_a) {
	HANDLE_ERROR(cudaFree(d_a));
}

void copyToDev(float* d_a, float* h_a, size_t n) {
	HANDLE_ERROR(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
}

void copyFromDev(float* h_a, float* d_a, size_t n) {
	HANDLE_ERROR(cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost));
}

void setDevMem(float* d_a, float val, size_t n) {
	HANDLE_ERROR(cudaMemset(d_a, val, n * sizeof(float)));
}
