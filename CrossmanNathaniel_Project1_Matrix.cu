
//@copyright     All rights are reserved, this code/project is not Open Source or Free
//@bug           None Documented     
//@author        Nathaniel Crossman (U00828694)
//@email		 crossman.4@wright.edu
//
//@Professor     Meilin Liu
//@Course_Number CS 4370/6370-90
//@date			 Thursday, September 26, 2019
//
//@project_name:
//				Task 1: Basic Matrix Addition 
//				Task 2: Basic Matrix Multiplication 



// System includes
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
// CUDA runtime
#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"

//#include <helper_functions.h>
//#include <helper_cuda.h>



/**
* Matrix Addtions (CUDA Kernel) on the device: C = A + B
* Matrix are all of the same size dim (2*2, 16*16)
*/
__global__ void add_matrix_gpu(int *devise_matrix_A, int *devise_matrix_B, int *devise_matrix_C, int width) {
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	int index = row * width + col;
	int p_value = 0;
	if ((row < width) && (col < width)) {
		p_value = devise_matrix_A[index] + devise_matrix_B[index];
		devise_matrix_C[index] = p_value;
	}

}

__global__ void matrixMulKernel(int* devise_matrix_A, int* devise_matrix_B, int * devise_matrix_C, int width)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	int i = 0, j = 0,index = 0;
	if ((row < width) && (col < width)){
		int p_value = 0;
		for (int k = 0; k < width; ++k) {
			i = row * width + k;
			j = k * width + col;
			p_value += devise_matrix_A[i] * devise_matrix_B[j];
		}
		index = row * width + col;
		devise_matrix_C[index] = p_value;
	}
}
// Kernel functions above

//--------------------------------------------------------------------------
void freeMemory_h(int* h_matrix_A, int* h_matrix_B, int*h_matrix_C, int *h_matrix_final_gpu);
void freeMemory_d(int* d_matrix_A, int* d_matrix_B, int*d_matrix_C);
void add_matrix_cpu(int *matrix_A, int *matrix_B, int *matrix_C, int width);

int verify(int *matrix_A, int *matrix_B, int width);
void addMatrixMain();
//Helper f
int menuShow();
void mainSwitch(int option);
void cls();
int debugOn();
void matrixMulOnHost(int* M, int* N, int* P, int width);
void multiplicationMatrixMain();
void initializationM(int *h_matrix_A, int *h_matrix_B, int width);
//helper for both parts
void getBlockSize(int &blockSize);
void getWidth(int &width);
void initialization(int *h_matrix_A, int *h_matrix_B, int width);
void printf_matrix(int *matrix_A, int width);

//Above is all prototypes

int main()
{	
	// This will pick the best possible CUDA capable device, otherwise
	// override the device ID based on input provided at the command line
	//int dev = findCudaDevice(argc, (const char **)argv);
	while (true) {
		mainSwitch(menuShow());
		printf("\n");
	}
	return 0;
}

int menuShow() {
	int hold;
	do {
		printf("1. Add Matrix \n");
		printf("2. Multiply Matrix \n");
		printf("3. Quit\n");
		printf("---------------------------------------\n");
		printf("Enter Choice: ");
		scanf("%d", &hold);

		if (hold < 1 || hold > 3) {
			cls();
		}
	} while (hold < 1 || hold > 3);
	return hold;
}
void cls() {
	for (int i = 0; i < 30; i++)
			printf("\n");
	system("@cls||clear");
}
/*
This function is like the driver function.
It hold the switch statement that called the function.
*/
void mainSwitch(int option) {
	switch (option) {
	case 1:
		addMatrixMain();
		break;
	case 2:
		multiplicationMatrixMain();
		break;
	case 3:
		exit(0);
		break;
	}
}

void getWidth(int &width) {
	printf("Please specify your square matrix dimension\n");
	printf("For example, you could enter 64 and the matrix dimension 64*64\n");
	printf("Enter Square Matrix size:");
	scanf("%d", &width);
	cls();
}
void getBlockSize(int &blockSize) {
	printf("Please specify your Block size \n");
	printf("For example, you could enter 4 and the block size would be 4 * 4 \n");
	printf("Enter Block Size:");
	scanf("%d", &blockSize);
	cls();
	
}
void initialization(int *h_matrix_A, int *h_matrix_B, int width) {
	int i = 0, j = 0, index = 0;
	int init = 1325;
	for (i = 0; i<width; ++i) {
		for (j = 0; j<width; ++j)
		{
			index = i * width + j;
			init = 3125 * init % 65536;
			h_matrix_A[index] = (init - 32768) / 16384;
			h_matrix_B[index] = init % 1000;
		}
	}
}
void initializationM(int *h_matrix_A, int *h_matrix_B, int width) {
	int i = 0, j = 0, index = 0;
	int init = 1325;
	for (i = 0; i<width; ++i) {
		for (j = 0; j<width; ++j)
		{
			index = i * width + j;
			init = 3125 * init % 65536;
			h_matrix_A[index] = (init - 32768) / 6553;
			h_matrix_B[index] = init % 1000;
		}
	}
}

/*
We are working on the host so we can use a 2D array like normal
*/
void add_matrix_cpu(int *matrix_A, int *matrix_B, int *matrix_C, int width) {
	int i, j, index;
	//pre-increment operator (++i) merely increments and returns. faster then i++
	for (i = 0; i < width; ++i) { //row = y
		for (j = 0; j < width; ++j) { //col = x
			index = i * width + j;
			//index = i + j * width;
			matrix_C[index] = matrix_A[index] + matrix_B[index];

		}
	}
}
void printf_matrix(int *matrix_A, int width) {
	int i, j, index;
	for (i = 0; i < width; ++i)
	{
		for (j = 0; j < width; ++j) {
			index = i * width + j;
			printf("%d \t", matrix_A[index]);
		}
		printf("\n");
	}
	printf("\n");
}
int verify(int *matrix_A, int *matrix_B, int width) {
	int index = 0;
	for (int i = 0; i < width; i++){
		for (int j = 0; j < width; j++){
			//index = i + j * width;
			index = i * width + j;

			if (matrix_A[index] != matrix_B[index]){
				printf("Test failed\n");
				return 0;
			}
		}
	}
	printf("The Test Passed\n");
	return 1;
}
void matrixMulOnHost(int* M, int* N, int* P, int width){
	for (int i = 0; i<width; i++)
	{
		for (int j = 0; j < width; ++j)
		{
			int sum = 0;
			for (int k = 0; k < width; ++k)
			{
				int a = M[i * width + k];
				int b = N[k * width + j];
				sum += a * b;
			}
			P[i * width + j] = sum;
		}
	}
}
int debugOn() {
	int hold;
	do {
		printf("\nRun in debug mode?\n");
		printf("Debug mode prints out alot of helpful info,\nbut it can takes a long time with big matrixes\n");
		printf("Enter 1 for Yes and 0 for No:");
		scanf("%d", &hold);
		if (hold < 0 || hold > 1) {
			cls();
		}
	} while (hold < 0 || hold > 1);
	cls();
	return hold;
}

void addMatrixMain() {
	int width = 0, blockSize = 0;
	int *h_matrix_A, *h_matrix_B, *h_matrix_C;
	int *d_matrix_A, *d_matrix_B, *d_matrix_C;
	int * h_matrix_final_gpu;
	int booleanValue = debugOn();

	getWidth(width);
	getBlockSize(blockSize);
	printf("Matrix Size: %d * %d \nSize of Thread block: %d * %d", width, width, blockSize, blockSize);
	printf("\n\n");
	printf("Adding...\n");
	//The size of all matrixes
	size_t dsize = (width * width) * sizeof(int);

	//Allocate memory for matrices on host
	h_matrix_A = (int*)malloc(dsize);
	h_matrix_B = (int*)malloc(dsize);
	h_matrix_C = (int*)malloc(dsize);
	h_matrix_final_gpu = (int*)malloc(dsize);

	//Set all matrices to 0 , Not needed but used for testing
	memset(h_matrix_A, 0, dsize);
	memset(h_matrix_B, 0, dsize);
	memset(h_matrix_C, 0, dsize);


	//Allocate memory for device Matrix
	cudaMalloc((void **)(&d_matrix_A), dsize);
	cudaMalloc((void **)(&d_matrix_B), dsize);
	cudaMalloc((void **)(&d_matrix_C), dsize);
	//checkCudaErrors(cudaMalloc((void **)(&d_matrix_A), dsize));
	//checkCudaErrors(cudaMalloc((void **)(&d_matrix_B), dsize));
	//checkCudaErrors(cudaMalloc((void **)(&d_matrix_C), dsize));


	initialization(h_matrix_A, h_matrix_B, width);

	add_matrix_cpu(h_matrix_A, h_matrix_B, h_matrix_C, width);
	if (booleanValue) {
		printf_matrix(h_matrix_A, width);
		printf_matrix(h_matrix_B, width);

		printf("\nThe results of CPU addition\n");
		printf_matrix(h_matrix_C, width);
	}

	//copy the Matrices from Host to Device
	cudaMemcpy(d_matrix_A, h_matrix_A, dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_B, h_matrix_B, dsize, cudaMemcpyHostToDevice);
	/*checkCudaErrors(cudaMemcpy(d_matrix_A, h_matrix_A, dsize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_matrix_B, h_matrix_B, dsize, cudaMemcpyHostToDevice));*/


	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(ceil(((double)width) / dimBlock.x), ceil(((double)width) / dimBlock.y));


	//GPU
	add_matrix_gpu << < dimGrid, dimBlock >> >(d_matrix_A, d_matrix_B, d_matrix_C, width);

	// Copy result from device to host
	cudaMemcpy(h_matrix_final_gpu, d_matrix_C, dsize, cudaMemcpyDeviceToHost);

	//checkCudaErrors(cudaMemcpy(h_matrix_final_gpu, d_matrix_C, dsize, cudaMemcpyHostToDevice));
	printf("GPU done Addition\n");

	if (booleanValue) {
		printf("\nThe results of GPU Addition\n");
		printf_matrix(h_matrix_final_gpu, width);
	}
	printf("\nVerifying\n");
	verify(h_matrix_C, h_matrix_final_gpu, width);

	// Clean up memory
	freeMemory_h(h_matrix_A, h_matrix_B, h_matrix_C, h_matrix_final_gpu);
	freeMemory_d(d_matrix_A, d_matrix_B, d_matrix_C);
}

void multiplicationMatrixMain() {
	int width = 0, blockSize = 0;
	int *h_matrix_A, *h_matrix_B, *h_matrix_C;
	int *d_matrix_A, *d_matrix_B, *d_matrix_C;
	int * h_matrix_final_gpu;
	int booleanValue = debugOn();
	getWidth(width);
	getBlockSize(blockSize);
	printf("Matrix Size: %d * %d \nSize of Thread block: %d * %d", width, width, blockSize, blockSize);
	printf("\n\n");
	printf("multiplying....\n");
	//The size of all matrixes
	size_t dsize = (width * width) * sizeof(int);

	//Allocate memory for matrices on host
	h_matrix_A = (int*)malloc(dsize);
	h_matrix_B = (int*)malloc(dsize);
	h_matrix_C = (int*)malloc(dsize);
	h_matrix_final_gpu = (int*)malloc(dsize);

	//Set all matrices to 0 , Not needed but used for testing
	memset(h_matrix_A, 0, dsize);
	memset(h_matrix_B, 0, dsize);
	memset(h_matrix_C, 0, dsize);


	//Allocate memory for device Matrix
	cudaMalloc((void **)(&d_matrix_A), dsize);
	cudaMalloc((void **)(&d_matrix_B), dsize);
	cudaMalloc((void **)(&d_matrix_C), dsize);
	/*checkCudaErrors(cudaMalloc((void **)(&d_matrix_A), dsize));
	checkCudaErrors(cudaMalloc((void **)(&d_matrix_B), dsize));
	checkCudaErrors(cudaMalloc((void **)(&d_matrix_C), dsize));*/


	initializationM(h_matrix_A, h_matrix_B, width);

	matrixMulOnHost(h_matrix_A, h_matrix_B, h_matrix_C, width);

	if (booleanValue) {
		printf_matrix(h_matrix_A, width);
		printf_matrix(h_matrix_B, width);

		printf("\nThe results of CPU Multiplication\n");
		printf_matrix(h_matrix_C, width);
	}
	//copy the Matrices from Host to Device
	cudaMemcpy(d_matrix_A, h_matrix_A, dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_B, h_matrix_B, dsize, cudaMemcpyHostToDevice);
	/*checkCudaErrors(cudaMemcpy(d_matrix_A, h_matrix_A, dsize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_matrix_B, h_matrix_B, dsize, cudaMemcpyHostToDevice));*/


	dim3 dimBlock(blockSize, blockSize);
	dim3 dimGrid(ceil(((double)width) / dimBlock.x), ceil(((double)width) / dimBlock.y));


	//GPU
	matrixMulKernel << < dimGrid, dimBlock >> >(d_matrix_A, d_matrix_B, d_matrix_C, width);

	// Copy result from device to host
	cudaMemcpy(h_matrix_final_gpu, d_matrix_C, dsize, cudaMemcpyDeviceToHost);
	//checkCudaErrors(cudaMemcpy(h_matrix_final_gpu, d_matrix_C, dsize, cudaMemcpyDeviceToHost));
	printf("GPU done Multiplying Matrixes\n");
	if (booleanValue) {
		printf("\nThe results of GPU Multiplication\n");
		printf_matrix(h_matrix_final_gpu, width);
	}
	printf("\nVerifying\n");
	verify(h_matrix_C, h_matrix_final_gpu, width);

	// Clean up memory
	freeMemory_h(h_matrix_A, h_matrix_B, h_matrix_C, h_matrix_final_gpu);
	freeMemory_d(d_matrix_A, d_matrix_B, d_matrix_C);
}

void freeMemory_h(int* h_matrix_A,int* h_matrix_B, int*h_matrix_C, int *h_matrix_final_gpu) {
	// Clean up memory
	free(h_matrix_A);
	free(h_matrix_B);
	free(h_matrix_C);
	free(h_matrix_final_gpu);

}

void freeMemory_d(int* d_matrix_A, int* d_matrix_B, int*d_matrix_C) {
	// Clean up memory
	cudaFree(d_matrix_A);
	cudaFree(d_matrix_B);
	cudaFree(d_matrix_C);
	//checkCudaErrors(cudaFree(d_matrix_A));
	//checkCudaErrors(cudaFree(d_matrix_B));
	//checkCudaErrors(cudaFree(d_matrix_C));
}