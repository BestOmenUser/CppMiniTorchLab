#include <cuda_runtime.h>
#include <device_functions.h> 
#include<curand_kernel.h>
#include <device_launch_parameters.h>
#include"stdio.h"
#include<iostream>
#include"cmath"
#include<vector>
#include "Tool.h"
#define NUMBERS_OF_THREADS 32
typedef std::vector<size_t> shape;

#if !defined (__CUDA_ARCH__) || __CUDA_ARCH__ >=600

#else
__device__ double atomicAdd(double* address, double val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return  __longlong_as_double(old);
}
#endif


__global__ void test(double* test) {
	atomicAdd(&test[0], 0.01);
}

extern "C" size_t GetLength(shape Shape) {
	size_t size = 1;
	for (auto i : Shape) {
		size *= i;
	}
	return size;
}

__global__ void SetValueGPU(double* X, double Start,size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		X[IDX] = Start+IDX+1.0;
	}

}
void SetValueCPU() {
}

void  MatrixOperator::SetValue(double* X, shape ShapeX,double ValueStart) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = ShapeX[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(ShapeX) / ShapeX[0];

		SetValueGPU << <numbers_of_block, numbers_of_thread >> > (X,ValueStart,threads);
		cudaDeviceSynchronize();
	}
	else {
		SetValueCPU();
	}
}

__global__ void SetNumberGPU(double* X, double value, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		X[IDX] = value;
	}

}
void SetNumberCPU() {
}

void  MatrixOperator::SetNumber(double* X, shape ShapeX, double Value) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = ShapeX[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(ShapeX) / ShapeX[0];

		SetNumberGPU << <numbers_of_block, numbers_of_thread >> > (X, Value, threads);
		cudaDeviceSynchronize();
	}
	else {
		SetNumberCPU();
	}
}

void  MatrixOperator::PrintValue(double* X, shape Shape) {
	size_t length = GetLength(Shape);
	for (size_t i = 0; i < length; i++) {
		std::cout << X[i] << " ";	
		if ((i + 1) % Shape[3] == 0) {
			printf("\n");
		}
	}
	std::cout << std::endl;
}

__global__  void MultiplyGpu2D(double* X, double* Y,double* Z,size_t threads,size_t Length)
{
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
		for (size_t i = idx; i < threads; i = i + stride) {
		double val = 0;
		for (size_t j = 0; j < Length; j++) {
			val += X[row * Length + j] * Y[j * threads + i];
		}
		Z[row * threads+ i] = val;
	}
}

void MultiplyCpu2D(double* X, shape ShapeX, double* Y, shape ShapeY, double* Z, shape ShapeZ) {
	size_t row = ShapeZ[0];
	size_t column = ShapeZ[3];
	size_t length = ShapeX[3];
	for (size_t i = 0; i < row; i++) {
		for (size_t j = 0; j < column; j++) {
			double val = 0;
			for (size_t k = 0; k < length; k++) {
				val+= X[i * length + k] * Y[k * column + j];
			}
			Z[i * column + j] = val;
		}
	}
}


void  MatrixOperator::Multiply2D(double* X, shape ShapeX, double* Y, shape ShapeY, double* Z, shape ShapeZ){
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
			size_t numbers_of_block = ShapeX[0];
			size_t length = ShapeX[3];
			size_t numbers_of_thread = NUMBERS_OF_THREADS;
			size_t threads = ShapeY[3];
			MultiplyGpu2D <<<numbers_of_block, numbers_of_thread >> > (X,Y,Z,threads,length);
			cudaDeviceSynchronize();
	}
	else {
		MultiplyCpu2D(X, ShapeX, Y, ShapeY, Z, ShapeZ);
	}
}


void  ConvolutionCPU(){}

__global__ void ConvolutionGPU(double* Input, double* Kernel, double* Output,
	size_t IS1,size_t IS2,size_t IS3,size_t IS4,
	size_t KS1,size_t KS2,size_t KS3,size_t KS4,
	size_t OS1,size_t OS2,size_t OS3,size_t OS4,
	size_t stride) 
{
	for (size_t thread = threadIdx.x; thread < OS2 * OS3 * OS4; thread += blockDim.x) {
		size_t O1 = blockIdx.x; //第几张图片
		size_t O2 = thread / (OS3 * OS4);  //第几个chanel
		size_t O3 = (thread % (OS3 * OS4)) / OS4; //一个chanel的第几个行
		size_t O4 = thread% OS4; //一个chanel的第几列
		double value = 0.0;
		size_t OutIDX = O1 * OS2 * OS3 * OS4 + thread;
		size_t INIDX = O1 * IS2 * IS3 * IS4;
		size_t InputX=O3*stride, InputY=O4*stride;
		for (size_t KernelSizeInput = 0; KernelSizeInput < KS2; KernelSizeInput++) {
			for (size_t KernelSizeWidth = 0; KernelSizeWidth < KS3; KernelSizeWidth++) {
				for (size_t KernelSizeHeight = 0; KernelSizeHeight < KS4; KernelSizeHeight++) {
					value += Input[INIDX + KernelSizeInput * IS3 * IS4+(InputX+KernelSizeWidth)*IS4+InputY+KernelSizeHeight]
						* Kernel[O2 * KS2 * KS3 * KS4 + KernelSizeInput * KS3 * KS4 + KernelSizeWidth * KS4 + KernelSizeHeight];
				}
			}

		}
		Output[OutIDX] = value;
	}
}

void MatrixOperator::Convolution(double* Input, shape InputShape, double* Kernel, shape KernelShape, double* Output, shape OutputShape, size_t stride)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = OutputShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		ConvolutionGPU << <numbers_of_block, numbers_of_thread >> > (Input,Kernel, Output,
			InputShape[0],InputShape[1],InputShape[2],InputShape[3],
			KernelShape[0],KernelShape[1],KernelShape[2],KernelShape[3],
			OutputShape[0],OutputShape[1],OutputShape[2],OutputShape[3],stride);
		cudaDeviceSynchronize();
	}
	else {
		ConvolutionCPU();
	}
}


__global__ void SetZeroGPU(double* X,size_t threads) {
	size_t row = blockIdx.x;
	size_t stride =blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		X[IDX] = 0.0;
	}
}
void SetZeroCPU(double *X, size_t size) {

	for (size_t i = 0; i < size; i++) {
		X[i] = 0;
	}
}

void  MatrixOperator::SetZero(double* X, shape ShapeX) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = ShapeX[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads =GetLength(ShapeX)/ShapeX[0];
		SetZeroGPU << <numbers_of_block, numbers_of_thread >> > (X,threads);
		cudaDeviceSynchronize();
	}
	else {
		SetZeroCPU(X,GetLength(ShapeX));
	}
}

__global__ void TransposeGpu2D(double* X, double* X_T,size_t row,size_t column) {
	size_t stride =  blockDim.x;
	size_t idx = threadIdx.x;
	for (size_t i = idx; i < column; i = i + stride) {
		X_T[i*row+blockIdx.x] = X[blockIdx.x * column + i];
	}
}

void TransposeCpu2D(double* X, shape& ShapeX, double* X_T) {
	size_t row = ShapeX[0];
	size_t column = ShapeX[3];
	for (size_t i = 0; i < row; i++) {
		for (size_t j = 0; j < column; j++) {
			X_T[j * row + i] = X[i * column + j];
		}
	}
}

void  MatrixOperator::Transpose(double* X, shape & ShapeX, double* X_T, shape & ShapeX_T)
{
	ShapeX_T = ShapeX;
	ShapeX_T[0] = ShapeX[3];
	ShapeX_T[3] = ShapeX[0];

	cudaError_t status = cudaSetDevice(0);
	size_t size = GetLength(ShapeX);
	if (status == cudaSuccess) {
		size_t numbers_of_threads = NUMBERS_OF_THREADS;
		size_t numbers_of_block = ShapeX[0];
		size_t row = ShapeX[0];
		size_t column = ShapeX[3];
		TransposeGpu2D <<<numbers_of_block, numbers_of_threads >>> (X,X_T,row,column);
		cudaDeviceSynchronize();
	}
	else {
		TransposeCpu2D(X, ShapeX, X_T);
	}
}



void MatrixAddCpu(double* X, double* Y, double* Z, shape& ShapeX, shape& ShapeY, shape &ShapeZ) {
}

__global__ void MatrixAddGpu(double* X, double* Y, double* Z,
	size_t X1,size_t X2,size_t X3,size_t X4,
	size_t Y1,size_t Y2,size_t Y3,size_t Y4,
	size_t Z1,size_t Z2,size_t Z3,size_t Z4,size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX% (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX% (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4)+
			O3%X3*X4+
			O4%X4;
		size_t idxY=O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		Z[IDX] = X[idxX] + Y[idxY];
	}
}



void  MatrixOperator::MatrixAdd(double* X, shape& ShapeX, double* Y, shape& ShapeY, double* Z, shape& ShapeZ) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		numbers_of_block = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ? ShapeX[0] : ShapeY[0];
		threads= (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ?
			GetLength(ShapeX)/ ShapeX[0] : GetLength(ShapeY)/ShapeY[0];
			ShapeZ = GetLength(ShapeX) >= GetLength(ShapeY) ? ShapeX : ShapeY;
			MatrixAddGpu << <numbers_of_block, numbers_of_thread >> >
				(X, Y, Z, ShapeX[0], ShapeX[1], ShapeX[2], ShapeX[3],
					ShapeY[0], ShapeY[1], ShapeY[2], ShapeY[3],
					ShapeZ[0], ShapeZ[1], ShapeZ[2], ShapeZ[3], threads);
		
		cudaDeviceSynchronize();
		}
	else {
		MatrixAddCpu(X, Y, Z, ShapeX, ShapeY, ShapeZ);
	}
}

void MatrixSubCpu(double* X, double* Y, double* Z, shape& ShapeX, shape& ShapeY, shape& ShapeZ) {
}

void  MatrixOperator::MatrixSub(double* X, shape& ShapeX, double* Y, shape& ShapeY, double* Z, shape& ShapeZ) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		double* CopyY;
		cudaMallocManaged(&CopyY, GetLength(ShapeY) * sizeof(double));
		MatrixNegative(Y, ShapeY, CopyY);
		numbers_of_block = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ? ShapeX[0] : ShapeY[0];
		threads = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ?
			GetLength(ShapeX) / ShapeX[0] : GetLength(ShapeY) / ShapeY[0];
			ShapeZ = GetLength(ShapeX) >= GetLength(ShapeY) ? ShapeX : ShapeY;
			MatrixAddGpu << <numbers_of_block, numbers_of_thread >> >
				(X, CopyY, Z, ShapeX[0], ShapeX[1], ShapeX[2], ShapeX[3],
					ShapeY[0], ShapeY[1], ShapeY[2], ShapeY[3],
					ShapeZ[0], ShapeZ[1], ShapeZ[2], ShapeZ[3], threads);
		cudaDeviceSynchronize();
		cudaFree(CopyY);
	}
	else {
		MatrixSubCpu(X, Y, Z, ShapeX, ShapeY, ShapeZ);
	}
}



void CopyCPU(){}

__global__ void CopyGPU(double* X,double* Y,size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i+ row * threads;
		Y[IDX] = X[IDX];
	}
}


void  MatrixOperator::Copy(double* FormData, shape Shape, double* TargetData) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape)/Shape[0];
		CopyGPU << <numbers_of_block, numbers_of_thread >> > (FormData,TargetData, threads);
		cudaDeviceSynchronize();
	}
	else {
		CopyCPU();
	}
}


void MultiplyNumberCPU() {}

__global__ void MultiplyNumberGPU(double* X, double* Y,double Number, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] =Number*X[IDX];
	}
}


void MatrixOperator::MultiplyNumber(double* FormData, shape Shape, double* TargetData, double Number)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		const unsigned int seed = 1234;
		MultiplyNumberGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData,Number, threads);
		cudaDeviceSynchronize();
	}
	else {
		MultiplyNumberCPU();
	}
}

void CompressVerticallyCPU(){}

__global__  void CompressVerticallyGPU(double* FormData, double* CompressedData,size_t CompressedSize,size_t size) {
	size_t idx = blockIdx.x;
	double value = 0.0;
	for (size_t i = blockIdx.x; i < size; i = i + CompressedSize) {
		value += FormData[i];
	}
	CompressedData[idx] = value;
}

void MatrixOperator::CompressVertically(double* FormData, shape FormShape, double* TargetData, shape TargetShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = GetLength(TargetShape);
		size_t numbers_of_thread = 1;
		CompressVerticallyGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, GetLength(TargetShape), GetLength(FormShape));
		cudaDeviceSynchronize();
	}
	else {
		CompressVerticallyCPU();
	}
}

void CompressHorizontallyCPU() {}

__global__  void CompressHorizontallyGPU(double* FormData, double* CompressedData, size_t CompressedSize, size_t size) {
	size_t idx = blockIdx.x;
	double value = 0.0;
	for (size_t i = blockIdx.x*CompressedSize; i < (blockIdx.x+1)* CompressedSize;  i++) {
		value += FormData[i];
	}
	CompressedData[idx] = value;
}

void MatrixOperator::CompressHorizontally(double* FormData, shape FormShape, double* TargetData, shape TargetShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = GetLength(TargetShape);
		size_t numbers_of_thread = 1;
		CompressHorizontallyGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, GetLength(FormShape)/GetLength(TargetShape), GetLength(FormShape));
		cudaDeviceSynchronize();
	}
	else {
		CompressHorizontallyCPU();
	}
}


void MatrixOperator::MatrixSum(double* Data, shape Shape, double& Output)
{
	size_t size = GetLength(Shape);
	double sum = 0;
	for (size_t i = 0; i < size; i++) {
		sum += Data[i];
	}
	Output = sum;
}

void MatrixSquareCPU(){}

__global__ void MatrixSquareGPU(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = X[IDX]*X[IDX];
	}
}

void MatrixOperator::MatrixSquare(double* FormData, shape Shape, double* TargetData)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		MatrixSquareGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, threads);
		cudaDeviceSynchronize();
	}
	else {
		MatrixSquareCPU();
	}
}


void MatrixNegativeCPU(){}


__global__  void MatrixNegativeGPU(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = -X[IDX];
	}
}

void MatrixOperator::MatrixNegative(double* FormData, shape Shape, double* TargetData)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		MatrixNegativeGPU<< <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, threads);
		cudaDeviceSynchronize();
	}
	else {
		MatrixNegativeCPU();
	}
}

void MatrixExpCPU(){}

__global__  void MatrixExpGPU(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = exp(X[IDX]);
	}
}

void MatrixOperator::MatrixExp(double* FormData, shape Shape, double* TargetData)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		MatrixExpGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, threads);
		cudaDeviceSynchronize();
	}
	else {
		MatrixExpCPU();
	}
}



void MatrixLogCPU() {}

__global__  void MatrixLogGPU(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = log(X[IDX]==0?1e-6: X[IDX]);
	}
}

void MatrixOperator::MatrixLog(double* FormData, shape Shape, double* TargetData)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		MatrixLogGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, threads);
		cudaDeviceSynchronize();
	}
	else {
		MatrixLogCPU();
	}
}


__global__ void MatrixDivisionGpu(double* X, double* Y, double* Z,
	size_t X1, size_t X2, size_t X3, size_t X4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t Z1, size_t Z2, size_t Z3, size_t Z4, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX % (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX % (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4) +
			O3 % X3 * X4 +
			O4 % X4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		Z[IDX] = X[idxX] / (Y[idxY] == 0 ? 1e-8 : Y[idxY]);
	}
}

void MatrixDivisionCpu(){}

void  MatrixOperator::MatrixDivision(double* X, shape& ShapeX, double* Y, shape& ShapeY, double* Z, shape& ShapeZ) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		numbers_of_block = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ? ShapeX[0] : ShapeY[0];
		threads = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ?
			GetLength(ShapeX) / ShapeX[0] : GetLength(ShapeY) / ShapeY[0];
		ShapeZ = GetLength(ShapeX) >= GetLength(ShapeY) ? ShapeX : ShapeY;
		MatrixDivisionGpu << <numbers_of_block, numbers_of_thread >> >
			(X, Y, Z, ShapeX[0], ShapeX[1], ShapeX[2], ShapeX[3],
				ShapeY[0], ShapeY[1], ShapeY[2], ShapeY[3],
				ShapeZ[0], ShapeZ[1], ShapeZ[2], ShapeZ[3], threads);
		cudaDeviceSynchronize();
	}
	else {
		MatrixDivisionCpu();
	}
}


void MatrixMultiplyCpu() {}

__global__ void MatrixMultiplyGpu(double* X, double* Y, double* Z,
	size_t X1, size_t X2, size_t X3, size_t X4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t Z1, size_t Z2, size_t Z3, size_t Z4, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX % (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX % (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4) +
			O3 % X3 * X4 +
			O4 % X4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		Z[IDX] = X[idxX] * Y[idxY];
	}
}

void  MatrixOperator::MatrixMultiply(double* X, shape& ShapeX, double* Y, shape& ShapeY, double* Z, shape& ShapeZ) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		numbers_of_block = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ? ShapeX[0] : ShapeY[0];
		threads = (GetLength(ShapeX) / GetLength(ShapeY) >= 1) ?
			GetLength(ShapeX) / ShapeX[0] : GetLength(ShapeY) / ShapeY[0];
		ShapeZ = GetLength(ShapeX) >= GetLength(ShapeY) ? ShapeX : ShapeY;
		MatrixMultiplyGpu << <numbers_of_block, numbers_of_thread >> >
				(X, Y, Z, ShapeX[0], ShapeX[1], ShapeX[2], ShapeX[3],
					ShapeY[0], ShapeY[1], ShapeY[2], ShapeY[3],
					ShapeZ[0], ShapeZ[1], ShapeZ[2], ShapeZ[3], threads);
		cudaDeviceSynchronize();
	}
	else {
		MatrixMultiplyCpu();
	}
}


void Multiply3DCPU() {}

__global__  void Multiply3DGPU(double* X, double* Y, double* Z,size_t N,size_t M,size_t P) {
	size_t row = blockIdx.x;
	size_t idx = threadIdx.x;
	size_t stride = blockDim.x;
	double value = 0;
	size_t idxX = row * N * M;
	size_t idxY = row * M * P;
	for (size_t i = idx; i < N * P; i=i+stride) {
		size_t idxX = row * N * M+(i/P)*N;
		size_t idxY = row * M * P+(i%P);
		value = 0.0;
		for (size_t j = 0; j < M; j++) {
			value += X[idxX + j] * Y[idxY + P * j];
		}
		Z[row * N * P + i] = value;
	}
}

void MatrixOperator::Multiply3D(double* X, shape ShapeX, double* Y, shape ShapeY, double* Z, shape ShapeZ)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = ShapeX[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		Multiply3DGPU << <numbers_of_block, numbers_of_thread >> >(X,Y,Z,ShapeX[2],ShapeX[3],ShapeY[3]);
		cudaDeviceSynchronize();
	}
	else {
		Multiply3DCPU();
	}
}


void MatrixGetLabelCPU() {}

__global__  void MatrixGetLabelGPU(double* FormData, double* CompressedData, size_t CompressedSize) {
	size_t idx = blockIdx.x;
	double value = 0.0;
	double label = -1.0;
	for (size_t i = blockIdx.x * CompressedSize; i < (blockIdx.x + 1) * CompressedSize; i++) {
		if (value < FormData[i]) {
			value = FormData[i];
			label = i% CompressedSize;
		}
	}
	CompressedData[idx] = label;
}


void MatrixOperator::MatrixGetLabel(double* FormData, shape FormShape, double* TargetData, shape TargetShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = GetLength(TargetShape);
		size_t numbers_of_thread = 1;
		MatrixGetLabelGPU << <numbers_of_block, numbers_of_thread >> > (FormData, TargetData, GetLength(FormShape) / GetLength(TargetShape));
		cudaDeviceSynchronize();
	}
	else {
		MatrixGetLabelCPU();
	}
}

void Rotation180CPU(){}

__global__ void Rotation180GPU(double* Former, double* Target, size_t IS1, size_t IS2, size_t IS3, size_t IS4) {
	for(size_t thread = threadIdx.x; thread < IS2 * IS3 * IS4; thread += blockDim.x) {
		size_t I1 = blockIdx.x;
		size_t I2 = thread / (IS3 * IS4);
		size_t I3 = thread % (IS3 * IS4) / IS4;
		size_t I4 = thread % IS4;
		size_t O1 = I2;
		size_t O2 = I1;
		size_t O3 = IS3 - I3-1;
		size_t O4 = IS4 - I4-1;
		size_t FormerIDX = I1 * IS2 * IS3 * IS4 + I2 * IS3 * IS4 + I3 * IS4 + I4;
		size_t TargetIDX = O1 * IS1 * IS3 * IS4 + O2 * IS3 * IS4 + O3 * IS4 + O4;
		Target[TargetIDX] = Former[FormerIDX];
	}
}

/*
旋转一个  M*N*X*Y后变为N*M*X*Y且X*Y块旋转180度,width,height
原来i,j,k,l位置变为j,i,k+1>width/2?width/2-(k+1-width/2):width/2+(width/2-k-1)=width-k-1,l=2*height-L
*/

void MatrixOperator::Rotation180(double* Former, shape FormerShape, double* Target, shape TargetShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block =FormerShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		Rotation180GPU << <numbers_of_block, numbers_of_thread >> > (Former, Target, FormerShape[0],
			FormerShape[1], FormerShape[2], FormerShape[3]);
		cudaDeviceSynchronize();
	}
	else {
		Rotation180CPU();
	}
}

void Compare(double* Label, double* Target, size_t size)
{
	std::cout << "Label     Predict" << std::endl;
	for (size_t i = 0; i < size;i++) {
		std::cout << Label[i] << "         " << Target[i] << std::endl;
	}
	std::cout << std::endl;
}

size_t GetMax(double* Data, size_t size)
{
	double max=-1;
	for (size_t i = 0; i < size;i++) {
		if (Data[i] > max) {
			max = Data[i];
		}
	}
	return (size_t)max;
}


void XavierCPU(){}

__global__ void XavierGPU(double* Data, size_t n_in, size_t n_out, unsigned int seed,size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX = 0;
	double stddev = sqrtf(2.0f / (n_in + n_out));
	curandState state;
	for(size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		curand_init(seed, IDX, 0, &state);
		Data[IDX]= curand_normal(&state) * stddev;
	}
}

void WeightInit::Xavier(double* Weight, shape Shape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block=Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads=GetLength(Shape)/Shape[0];
		const unsigned int seed = 1234;
		XavierGPU << <numbers_of_block, numbers_of_thread >> > (Weight, Shape[0], Shape[3], seed, threads);
		cudaDeviceSynchronize();
	}
	else {
		XavierCPU();
	}
}


void KaimingCPU() {}

__global__ void KaimingGPU(double* Data, size_t n_in, unsigned int seed, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX = 0;
	double stddev = sqrtf(2.0f /n_in);
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		curandState state;
		curand_init(seed, IDX, 0, &state);
		Data[IDX] = curand_normal(&state) * stddev;
	}
}

void WeightInit::Kaiming(double* Weight, shape Shape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape)/Shape[0];
		const unsigned int seed = 1234;
		KaimingGPU << <numbers_of_block, numbers_of_thread >> > (Weight, Shape[0], seed, threads);
		cudaDeviceSynchronize();
	}
	else {
		KaimingCPU();
	}
}


void SetOneCPU(){}
__global__ void SetOneGPU(double* Data,size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Data[IDX] = 1;
	}
}
void WeightInit::SetOne(double* Weight, shape Shape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape)/Shape[0];
		SetOneGPU << <numbers_of_block, numbers_of_thread >> > (Weight,threads);
		cudaDeviceSynchronize();
	}
	else {
		SetOneCPU();
	}
}




void ReLUForwardCPU() {}

__global__ void ReLUForwardGPU(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = X[IDX]>0.0?X[IDX]:0.0;
	}
}


void ReLUOperator::ReLUForward(double* Input, shape Shape, double* Output) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		ReLUForwardGPU << <numbers_of_block, numbers_of_thread >> > (Input, Output, threads);
		cudaDeviceSynchronize();
	}
	else {
		ReLUForwardCPU();
	}
}


void ReLUBackPropagationCPU() {}

__global__ void ReLUBackPropagationGPU(double* Input,double* Gradient,double* ReLUGradient, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		ReLUGradient[IDX] = Input[IDX] > 0.0 ?Gradient[IDX] : 0.0;
	}
}

void ReLUOperator::ReLUBackward(double* Input,double* Gradient, shape Shape, double* ReLUGradient)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		ReLUBackPropagationGPU << <numbers_of_block, numbers_of_thread >> > (Input,Gradient,ReLUGradient, threads);
		cudaDeviceSynchronize();
	}
	else {
		ReLUBackPropagationCPU();
	}
}


void  SoftMaxForwardCPU(){}

void SoftMaxOperator::SoftMaxForward(double* Input, shape Shape, double* Output) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		double  *CompressedExp;
		MatrixOperator::MatrixExp(Input, Shape,Output);
		shape CompressedShape = Shape;
		CompressedShape[3] = 1;
		cudaMallocManaged(&CompressedExp,sizeof(double) * GetLength(CompressedShape));
		MatrixOperator::CompressHorizontally(Output, Shape, CompressedExp, CompressedShape);
		MatrixOperator::MatrixDivision(Output, Shape, CompressedExp, CompressedShape,Output,Shape);
		cudaFree(CompressedExp);
	}
	else {
		SoftMaxForwardCPU();
	}
}


void JacobianCPU(){}

__global__ void JacobianGPU(double* Output, double* Jacobian,size_t threads,size_t rows) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	for (size_t i = idx; i < threads; i = i + stride) {
		Jacobian[row * threads + i] = (i / rows == i % rows) ?
			Output[row * rows + i % rows] * (1 - Output[row * rows + i / rows]) :
			-Output[row * rows + i % rows] * Output[row * rows + i / rows];
	}

}

void SoftMaxOperator::JacobianMatrix(double* Output, shape OutputShape, double* Jacobian, shape JacobianShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = JacobianShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(JacobianShape)/JacobianShape[0];
		JacobianGPU<< <numbers_of_block, numbers_of_thread >> > (Output,Jacobian, threads, JacobianShape[3]);
		cudaDeviceSynchronize();
	}
	else{
		JacobianCPU();
	}
}




void OneHotCpu(){}

__global__ void OneHotGpu(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = (IDX%threads == (size_t)X[row]) ? 1.0 : 0.0;
	}
}

void CrossEntropyOperator::OneHot(double* Input, shape InputShape, double* OneHot, shape OneHotShape) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = OneHotShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(OneHotShape) / OneHotShape[0];
		OneHotGpu<< <numbers_of_block, numbers_of_thread >> > (Input, OneHot, threads);
		cudaDeviceSynchronize();
	}
	else {
		ReLUForwardCPU();
	}
}


void SigmoidForwardCPU() {}

__global__ void SigmoidForwardGPU(double* X, double* Y, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		Y[IDX] = 1 / (1 + exp(-X[IDX]));
	}
}

void SigmoidOperator::SigmoidForward(double* Input, shape Shape, double* Output) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		SigmoidForwardGPU << <numbers_of_block, numbers_of_thread >> > (Input, Output, threads);
		cudaDeviceSynchronize();
	}
	else {
		SigmoidForwardCPU();
	}
}



void SigmoidBackPropagationCPU() {}

__global__ void SigmoidBackPropagationGPU(double* Output, double* Gradient, double* SigmoidGradient, size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		SigmoidGradient[IDX] =Gradient[IDX]*Output[IDX]*(1-Output[IDX]);
	}
}

void SigmoidOperator::SigmoidBackward(double* Output, double* Gradient, shape Shape, double* SigmoidGradient)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		SigmoidBackPropagationGPU << <numbers_of_block, numbers_of_thread >> > (Output, Gradient, SigmoidGradient, threads);
		cudaDeviceSynchronize();
	}
	else {
		SigmoidBackPropagationCPU();
	}
}


void PaddingCPU(){}

__global__ void PaddingGPU(double* Input,double* Output,
	size_t IS2,size_t IS3,size_t IS4,size_t OS2,size_t OS3,size_t OS4,size_t padding1,size_t padding2) 
{
	for (size_t thread = threadIdx.x; thread < IS2 * IS3 * IS4; thread += blockDim.x) {
		size_t I1 = blockIdx.x;
		size_t I2 = thread / (IS3 * IS4);
		size_t I3 = (thread % (IS3 * IS4)) / IS4;
		size_t I4 = thread %IS4;	
		size_t O3 = I3 + padding1;
		size_t O4 = I4 + padding2;
		size_t InputIDX = I1 * (IS2 * IS3 * IS4) + I2 * (IS3 * IS4) + I3 * IS4 + I4;
		size_t OutputIDX = I1 * (OS2 * OS3 * OS4) + I2 * (OS3 * OS4) + O3 * OS4 + O4;
		Output[OutputIDX] = Input[InputIDX];
	}
}

void ConvOperator::PaddingData(double* Input, shape ShapeInput, double* Output, shape ShapeOutput,size_t Padding1,size_t Padding2) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = ShapeInput[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(ShapeInput) / ShapeInput[0];
		PaddingGPU << <numbers_of_block, numbers_of_thread >> > (Input,Output, 
			ShapeInput[1],ShapeInput[2],ShapeInput[3],ShapeOutput[1],ShapeOutput[2],ShapeOutput[3],Padding1,Padding2);
		cudaDeviceSynchronize();
	}
	else {
		PaddingCPU();
	}
}


void ConvDLDKCPU(){}

__global__ void ConvDLDKGPU(double* Output, double* Input, double* KernelTemp,
size_t OS1,size_t OS2,size_t OS3,size_t OS4,
size_t IS1,size_t IS2,size_t IS3,size_t IS4,
size_t KS1,size_t KS2,size_t KS3,size_t KS4,size_t Stride)
{
	for (size_t thread = threadIdx.x; thread < KS2 * KS3 * KS4; thread += blockDim.x) {
		size_t K1 = blockIdx.x;
		size_t K2 = thread / (KS3 * KS4);
		size_t K3 = thread % (KS3 * KS4) / KS4;
		size_t K4 = thread % KS4;
		size_t KernelIDX = K1 * KS2 * KS3 * KS4 + thread;

		double value = 0.0;
		for (size_t ImgIDX = 0; ImgIDX < IS1; ImgIDX++) {
			for (size_t OutWidthIDX = 0; OutWidthIDX < OS3; OutWidthIDX++) {
				for (size_t OutHeightIDX = 0; OutHeightIDX < OS4; OutHeightIDX++) {
					size_t InputIDX = ImgIDX * IS2 * IS3 * IS4 + K2 * IS3 * IS4 + (K3 + OutWidthIDX * Stride) * IS4 + K4 + OutHeightIDX * Stride;
					size_t OutputIDX = ImgIDX * OS2 * OS3 * OS4 + K1 * OS3 * OS4 + OutWidthIDX * OS4 + OutHeightIDX;
					value += Input[InputIDX] * Output[OutputIDX];
				}
			}
		}

		KernelTemp[KernelIDX] = value;
	}
}

void ConvOperator::ConvDLDK(double* Output, shape OutputShape, double* Input, shape InputShape, double* KernelTemp, shape KernelShape, size_t stride)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = KernelShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(KernelShape) / KernelShape[0];
		ConvDLDKGPU<<<numbers_of_block,numbers_of_thread>>>(Output, Input, KernelTemp, OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3],
			InputShape[0], InputShape[1], InputShape[2], InputShape[3],
			KernelShape[0], KernelShape[1], KernelShape[2], KernelShape[3], stride
		);
		cudaDeviceSynchronize();
	}
	else {
		ConvDLDKCPU();
	}
}




void RecoveryPaddedCPU() {}

__global__ void RecoveryPaddedGPU(double* Recovery, double* Padded,
	size_t IS2, size_t IS3, size_t IS4, size_t OS2, size_t OS3, size_t OS4, size_t padding1, size_t padding2)
{
	for (size_t thread = threadIdx.x; thread < IS2 * IS3 * IS4; thread += blockDim.x) {
		size_t I1 = blockIdx.x;
		size_t I2 = thread / (IS3 * IS4);
		size_t I3 = (thread % (IS3 * IS4)) / IS4;
		size_t I4 = thread % IS4;
		size_t O3 = I3 + padding1;
		size_t O4 = I4 + padding2;
		size_t RecoveryIDX = I1 * (IS2 * IS3 * IS4) + I2 * (IS3 * IS4) + I3 * IS4 + I4;
		size_t PaddedIDX = I1 * (OS2 * OS3 * OS4) + I2 * (OS3 * OS4) + O3 * OS4 + O4;
		Recovery[RecoveryIDX] =Padded[PaddedIDX];
	}
}

void ConvOperator::RecoveryPadding(double* PaddedData, shape PaddedShape, double* RecoveryData, shape RecoverShape, size_t PaddingHorizontally, size_t PaddingVertically)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block =RecoverShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(RecoverShape) / RecoverShape[0];
		RecoveryPaddedGPU << <numbers_of_block, numbers_of_thread >> > (RecoveryData,PaddedData,
			RecoverShape[1],RecoverShape[2],RecoverShape[3],
			PaddedShape[1],PaddedShape[2],PaddedShape[3],PaddingHorizontally,PaddingVertically);
		cudaDeviceSynchronize();
	}
	else {
		RecoveryPaddedCPU();
	}
}





void  MaxPoolingConvolutionCPU() {}

__global__ void MaxPoolingConvolutionGPU(double* Input, double* Kernel, double* Output,
	size_t IS1, size_t IS2, size_t IS3, size_t IS4,
	size_t KS1, size_t KS2, size_t KS3, size_t KS4,
	size_t OS1, size_t OS2, size_t OS3, size_t OS4,
	size_t stride)
{
	for (size_t thread = threadIdx.x; thread < OS2 * OS3 * OS4; thread += blockDim.x) {
		size_t O1 = blockIdx.x; //第几张图片
		size_t O2 = thread / (OS3 * OS4);  //第几个chanel
		size_t O3 = (thread % (OS3 * OS4)) / OS4; //一个chanel的第几个行
		size_t O4 = thread % OS4; //一个chanel的第几列
		double value = 0.0;
		size_t OutIDX = O1 * OS2 * OS3 * OS4 + thread;
		size_t INIDX = O1 * IS2 * IS3 * IS4+O2*IS3*IS4;
		size_t InputX = O3 * stride, InputY = O4 * stride;
		value = Input[INIDX + InputX * IS4 + InputY];
		for (size_t KernelSizeWidth = 0; KernelSizeWidth < KS3; KernelSizeWidth++) {
				for (size_t KernelSizeHeight = 0; KernelSizeHeight < KS4; KernelSizeHeight++) {
					value = value >= Input[INIDX+ (InputX + KernelSizeWidth) * IS4 + InputY + KernelSizeHeight] ?
						value : Input[INIDX+ (InputX + KernelSizeWidth) * IS4 + InputY + KernelSizeHeight];						
				}
			}
		Output[OutIDX] = value;
	}
}

void PoolingOperator::MaxPoolingConv(double* Input, shape InputShape, double* Kernel, shape KernelShape, double* Output, shape OutputShape, size_t Stride)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = OutputShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		MaxPoolingConvolutionGPU << <numbers_of_block, numbers_of_thread >> > (Input, Kernel, Output,
			InputShape[0], InputShape[1], InputShape[2], InputShape[3],
			KernelShape[0], KernelShape[1], KernelShape[2], KernelShape[3],
			OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3], Stride);
		cudaDeviceSynchronize();
	}
	else {
		MaxPoolingConvolutionCPU();
	}
}


void MaxPoolingBackwardCPU(){}

__global__ void MaxPoolingBackwardGPU(double* Input,double* Output,double* Gradient,double* MaxPoolingGradient,
	size_t IS1,size_t IS2,size_t IS3,size_t IS4,
	size_t GS1,size_t GS2,size_t GS3,size_t GS4,
	size_t KernelWidth, size_t KernelHeight, size_t stride) {
	for (size_t thread = threadIdx.x; thread < GS2 * GS3 * GS4; thread += blockDim.x) {
		size_t G1 = blockIdx.x; //第几张图片
		size_t G2 = thread / (GS3 * GS4);  //第几个chanel
		size_t G3 = (thread % (GS3 * GS4)) / GS4; //一个chanel的第几个行
		size_t G4 = thread % GS4; //一个chanel的第几列
		size_t I1 = G1;
		size_t I2 = G2;
		size_t I3 = G3 * stride;
		size_t I4 = G4 * stride;
		size_t GradientIDX = G1 * GS2 * GS3 * GS4+thread;
		size_t InputIDX = I1 * IS2 * IS3 * IS4 + I2 * IS3 * IS4+I3*IS4+I4;
		for (size_t row = 0; row < KernelWidth; row++) {
			size_t nums = 0;
			for (size_t column = 0; column < KernelHeight; column++) {
				size_t IDX = InputIDX + row * IS4 + column;
				if (Input[IDX] == Output[GradientIDX]) {
					nums++;
				}
			}
			for (size_t column = 0; column < KernelHeight; column++) {
				size_t IDX = InputIDX + row * IS4 + column;
				if (Input[IDX] == Output[GradientIDX]) {
					atomicAdd(&MaxPoolingGradient[IDX], (Gradient[GradientIDX] / nums));
				}
			}
		}

	}
}

void PoolingOperator::MaxPoolingBackward(double* Input, shape InputShape, double* Output,double* Gradient, 
	shape GradientShape,double* MaxPoolingGradient, size_t KernelWidth, size_t KernelHeight, size_t Stride)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = GradientShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		MaxPoolingBackwardGPU << <numbers_of_block, numbers_of_thread >> > (Input,Output,Gradient,MaxPoolingGradient,
			InputShape[0], InputShape[1], InputShape[2], InputShape[3],
			GradientShape[0], GradientShape[1], GradientShape[2], GradientShape[3],
			KernelWidth, KernelHeight,Stride);
		cudaDeviceSynchronize();
	}
	else {
		MaxPoolingBackwardCPU();
	}
}

void AvgPoolingBackwardCPU() {}

__global__ void AvgPoolingBackwardGPU(double* Input, double* Output, double* Gradient, double* MaxPoolingGradient,
	size_t IS1, size_t IS2, size_t IS3, size_t IS4,
	size_t GS1, size_t GS2, size_t GS3, size_t GS4,
	size_t KernelWidth, size_t KernelHeight, size_t stride) {
	for (size_t thread = threadIdx.x; thread < GS2 * GS3 * GS4; thread += blockDim.x) {
		size_t G1 = blockIdx.x; //第几张图片
		size_t G2 = thread / (GS3 * GS4);  //第几个chanel
		size_t G3 = (thread % (GS3 * GS4)) / GS4; //一个chanel的第几个行
		size_t G4 = thread % GS4; //一个chanel的第几列
		size_t I1 = G1;
		size_t I2 = G2;
		size_t I3 = G3 * stride;
		size_t I4 = G4 * stride;
		size_t GradientIDX = G1 * GS2 * GS3 * GS4 + thread;
		size_t InputIDX = I1 * IS2 * IS3 * IS4 + I2 * IS3 * IS4 + I3 * IS4 + I4;
		for (size_t row = 0; row < KernelWidth; row++) {
			for (size_t column = 0; column < KernelHeight; column++) {
				size_t IDX = InputIDX + row * IS4 + column;
					atomicAdd(&MaxPoolingGradient[IDX], (Gradient[GradientIDX]/(KernelHeight*KernelWidth)));
				}
			}
		}
}

void PoolingOperator::AveragePoolingBackward(double* Input, shape InputShape, double* Output, double* Gradient, shape GradientShape, double* MaxPoolingGradient, size_t KernelWidth, size_t KernelHeight, size_t Stride)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = GradientShape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		AvgPoolingBackwardGPU << <numbers_of_block, numbers_of_thread >> > (Input, Output, Gradient, MaxPoolingGradient,
			InputShape[0], InputShape[1], InputShape[2], InputShape[3],
			GradientShape[0], GradientShape[1], GradientShape[2], GradientShape[3],
			KernelWidth, KernelHeight, Stride);
		cudaDeviceSynchronize();
	}
	else {
		AvgPoolingBackwardCPU();
	}
}


void DropOutForwardCPU() {}

__global__ void DropOutForwardGPU(double* X, double* Y,double Rate,unsigned int seed,size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	curandState state;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		curand_init(seed, IDX, 0, &state);
		double random_value = curand_uniform(&state);
		Y[IDX] = random_value>Rate? X[IDX] : 0.0;
	}
}

void DropOutOperator::DropOutForward(double* Input, shape Shape, double* Output, double Rate) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		unsigned int seed = 1234;
		DropOutForwardGPU << <numbers_of_block, numbers_of_thread >> > (Input, Output,Rate,seed, threads);
		cudaDeviceSynchronize();
	}
	else {
		DropOutForwardCPU();
	}
}



void DropOutBackPropagationCPU() {}

__global__ void DropOutBackPropagationGPU(double* Input,double* Output, double* Gradient, double* DropOutGradient,double Rate,size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		DropOutGradient[IDX] = (Output[IDX] ==Output[IDX] )? Gradient[IDX] / Rate : 0.0;
	}
}


void DropOutOperator::DropOutBackward(double* Input,double*Output, double* Gradient, shape Shape, double* DropOutGradient, double Rate)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		DropOutBackPropagationGPU << <numbers_of_block, numbers_of_thread >> > (Input,Output, Gradient, DropOutGradient,Rate, threads);
		cudaDeviceSynchronize();
	}
	else {
		DropOutBackPropagationCPU();
	}
}


void L1BackPropagationCPU() {}

__global__ void L1BackPropagationGPU(double* Weight,double* NormBP,double LambDa,size_t threads) {
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t idx = threadIdx.x;
	size_t IDX = 0;
	for (size_t i = idx; i < threads; i = i + stride) {
		IDX = i + row * threads;
		NormBP[IDX] =Weight[IDX]>0 ? LambDa : -LambDa;
	}
}

void L1Operator::L1Backward(double* Weight, shape Shape, double* NormBP, double Lambda) {
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block = Shape[0];
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads = GetLength(Shape) / Shape[0];
		L1BackPropagationGPU << <numbers_of_block, numbers_of_thread >> > (Weight, NormBP, Lambda, threads);
		cudaDeviceSynchronize();
	}
	else {
		L1BackPropagationCPU();
	}
}


void BatchNormOperator::CalculateMean(double* Input, shape InputShape, double* Mean, shape MeanShape) {
	double* Temp;
	shape TempShape = InputShape;
	TempShape[0] = 1;
	cudaMallocManaged(&Temp, GetLength(TempShape));
	MatrixOperator::CompressVertically(Input, InputShape, Temp, TempShape);
	MatrixOperator::CompressHorizontally(Temp, TempShape, Mean, MeanShape);
	size_t size = GetLength(InputShape) / GetLength(MeanShape);
	double* number;
	shape numberShape = { 1,1,1,1 };
	cudaMallocManaged(&number,sizeof(double)*GetLength(numberShape));
	*number= (double)size;
	MatrixOperator::MatrixDivision(Mean, MeanShape, number, numberShape, Mean, MeanShape);
	cudaFree(Temp);
	cudaFree(number);
}

void BatchNormOperator::CalculateVariance(double* Input, shape InputShape, double* Mean, shape MeanShape, double* Variance, shape VarianceShape)
{
	double* SubTemp;
	shape SubShape = InputShape;
	cudaMallocManaged(&SubTemp, GetLength(SubShape) * sizeof(double));
	MatrixOperator::MatrixSub(Input, InputShape, Mean, MeanShape, SubTemp, SubShape);
	MatrixOperator::MatrixMultiply(SubTemp, SubShape, SubTemp, SubShape, SubTemp, SubShape);
	CalculateMean(SubTemp, SubShape, Variance, VarianceShape);
	cudaFree(SubTemp);
}

void BatchNormOperator::CalculateMoving(double* Moving, shape MovingShape, double* NewValue, shape NewValueShape,double* NewMoving,double Momentum)
{
	double* TempMoving;
	double* TempValue;
	cudaMallocManaged(&TempMoving, GetLength(MovingShape)*sizeof(double));
	cudaMallocManaged(&TempValue, GetLength(MovingShape) * sizeof(double));
	MatrixOperator::MultiplyNumber(Moving, MovingShape, TempMoving, Momentum);
	MatrixOperator::MultiplyNumber(NewValue, MovingShape, TempValue,1-Momentum);
	MatrixOperator::MatrixAdd(TempMoving, MovingShape, TempValue, MovingShape, NewMoving, MovingShape);
	cudaFree(TempMoving);
	cudaFree(TempValue);
}



void CalculateValueHatCpu() {}

__global__ void CalculateValueHatGpu(double* Input, double* Mean, double* Variance,double* Output,
	size_t IS1, size_t IS2, size_t IS3, size_t IS4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t OS1, size_t OS2, size_t OS3, size_t OS4, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (OS2 * OS3 * OS4);
		size_t O2 = (IDX % (OS2 * OS3 * OS4)) / (OS3 * OS4);
		size_t O3 = (IDX % (OS3 * OS4) / OS4);
		size_t O4 = IDX % OS4;

		size_t idxX = O1 % IS1 * (IS2 * IS3 * IS4) +
			O2 % IS2 * (IS3 * IS4) +
			O3 % IS3 * IS4 +
			O4 % IS4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		Output[IDX] = (Input[idxX]-Mean[idxY])/sqrt(Variance[idxY]+1e-5);
	}
}


void BatchNormOperator::CalculateValueHat(double* Value, shape ValueShape, double* Mean, shape MeanShape, double* Variance, shape VarianceShape, double* ValueHat)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		numbers_of_block = (GetLength(ValueShape) / GetLength(MeanShape) >= 1) ? ValueShape[0] : MeanShape[0];
		threads = (GetLength(ValueShape) / GetLength(MeanShape) >= 1) ?
			GetLength(ValueShape) / ValueShape[0] : GetLength(MeanShape) / MeanShape[0];
		CalculateValueHatGpu << <numbers_of_block, numbers_of_thread >> >
			(Value, Mean, Variance,ValueHat,ValueShape[0], ValueShape[1], ValueShape[2], ValueShape[3],
				MeanShape[0], MeanShape[1], MeanShape[2], MeanShape[3],
				ValueShape[0], ValueShape[1], ValueShape[2], ValueShape[3], threads);

		cudaDeviceSynchronize();
	}
	else {
		CalculateValueHatCpu();
	}
}


void NormalizeBackwardCPU(){}

__global__ void NormalizeBackwardGPU(double* Input, double* Gradient, double* Mean, double* Variance,
	double* MeanGradient, double* VarianceGradient, double* BatchGradient,double* Scale,double* Offset,
	size_t IS1, size_t IS2, size_t IS3, size_t IS4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t OS1, size_t OS2, size_t OS3, size_t OS4,size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (OS2 * OS3 * OS4);
		size_t O2 = (IDX % (OS2 * OS3 * OS4)) / (OS3 * OS4);
		size_t O3 = (IDX % (OS3 * OS4) / OS4);
		size_t O4 = IDX % OS4;

		size_t idxX = O1 % IS1 * (IS2 * IS3 * IS4) +
			O2 % IS2 * (IS3 * IS4) +
			O3 % IS3 * IS4 +
			O4 % IS4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		double  DLDXHat = Gradient[idxX] * Scale[idxY];
		BatchGradient[IDX] = DLDXHat / (sqrt(Variance[idxY] + 1e-5)) +
			2.0 / (IS1 * IS3 * IS4) * (Input[idxX] - Mean[idxY]) * VarianceGradient[idxY] + MeanGradient[idxY] / (IS1 * IS3 * IS4);
	}
}




void  BatchNormOperator::NormalizeBackward(double* Input, double* InputHat, double* Output, double* Gradient, double* BatchGradient,shape InputShape,
	double* Mean, double* Variance, double* Scale, double* Offset, double* ScaleGradient,
	double* OffsetGradient, shape MeanShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		double* MeanGradient;
		double* VarianceGradient;
		cudaMallocManaged(&MeanGradient, sizeof(double) * GetLength(MeanShape));
		cudaMallocManaged(&VarianceGradient, sizeof(double) * GetLength(MeanShape));
		BatchNormOperator::GetDLDVariance(Input, InputShape, Gradient, Mean, Variance, VarianceGradient,MeanShape, Scale);
		BatchNormOperator::GetDLDMean(Input, InputShape, Gradient, Mean, Variance, MeanGradient, MeanShape, Scale);
		BatchNormOperator::GetDLDScale(InputHat, Gradient, InputShape, Scale, ScaleGradient, MeanShape);
		BatchNormOperator::GetDLDOffset(Gradient, InputShape, Scale, OffsetGradient, MeanShape);

		numbers_of_block = (GetLength(InputShape) / GetLength(MeanShape) >= 1) ? InputShape[0] : MeanShape[0];
		threads = (GetLength(InputShape) / GetLength(MeanShape) >= 1) ?
			GetLength(InputShape) / InputShape[0] : GetLength(MeanShape) / MeanShape[0];

		NormalizeBackwardGPU << <numbers_of_block, numbers_of_thread >> >
			(Input, Gradient, Mean, Variance, MeanGradient, VarianceGradient, BatchGradient,Scale,Offset,
				InputShape[0], InputShape[1], InputShape[2], InputShape[3],
				MeanShape[0],MeanShape[1],MeanShape[2],MeanShape[3],
				InputShape[0], InputShape[1], InputShape[2], InputShape[3],
				threads);
		cudaDeviceSynchronize();
		cudaFree(MeanGradient);
		cudaFree(VarianceGradient);
	}
	else {
		NormalizeBackwardCPU();
	}
}


void DLDVarianceCpu(){}

__global__ void DLDVarianceGpu(double* Input, double* Gradient, double* Mean, double* Variance, double* Output,
	size_t X1, size_t X2, size_t X3, size_t X4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t Z1, size_t Z2, size_t Z3, size_t Z4,double* Scale, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX % (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX % (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4) +
			O3 % X3 * X4 +
			O4 % X4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		Output[IDX] = Scale[idxY] * Gradient[idxX] * (Input[idxX] - Mean[idxY]) * -0.5 * pow( (Variance[idxY] + 1e-5), -1.5);
	}
}


void BatchNormOperator::GetDLDVariance(double* Input, shape OutputShape,
	double* Gradient, double* Mean, double* Variance, double* VarianceGradient, shape  VarianceGradientShape, double* Scale)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		double* Temp;
		shape ShapeZ=OutputShape;
		cudaMallocManaged(&Temp, sizeof(double) * GetLength(OutputShape));
		numbers_of_block = (GetLength(OutputShape) / GetLength(VarianceGradientShape) >= 1) ? OutputShape[0] : VarianceGradientShape[0];
		threads = (GetLength(OutputShape) / GetLength(VarianceGradientShape) >= 1) ?
			GetLength(OutputShape) / OutputShape[0] : GetLength(VarianceGradientShape) / VarianceGradientShape[0];
		DLDVarianceGpu << <numbers_of_block, numbers_of_thread >> >
			(Input,Gradient, Mean,Variance,Temp, OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3],
				VarianceGradientShape[0], VarianceGradientShape[1], VarianceGradientShape[2], VarianceGradientShape[3],
				ShapeZ[0], ShapeZ[1], ShapeZ[2], ShapeZ[3],Scale, threads);
		cudaDeviceSynchronize();
		double* Compressed;
		shape CompressedShape = OutputShape;
		CompressedShape[0] = 1;
		cudaMallocManaged(&Compressed, sizeof(double) * GetLength(CompressedShape));
		MatrixOperator::CompressVertically(Temp, OutputShape, Compressed, CompressedShape);
		MatrixOperator::CompressHorizontally(Compressed, CompressedShape, VarianceGradient, VarianceGradientShape);
		cudaFree(Temp);
		cudaFree(Compressed);
	}
	else {
		DLDVarianceCpu();
	}
}




void DLDMeanCpu() {}

__global__ void DLDMeanGpu(double* Input, double* Gradient, double* Mean, double* Variance, double* Output,
	size_t X1, size_t X2, size_t X3, size_t X4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t Z1, size_t Z2, size_t Z3, size_t Z4, double* Scale, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX % (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX % (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4) +
			O3 % X3 * X4 +
			O4 % X4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		Output[IDX] = Scale[idxY] * Gradient[idxX]*( - pow(0.5 * (Variance[idxY] + 1e-5), -0.5));
	}
}


void BatchNormOperator::GetDLDMean(double* Input, shape OutputShape,
	double* Gradient, double* Mean, double* Variance, double* MeanGradient, shape  MeanGradientShape, double* Scale)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		double* Temp;
		shape ShapeZ = OutputShape;
		cudaMallocManaged(&Temp, sizeof(double) * GetLength(OutputShape));
		numbers_of_block = (GetLength(OutputShape) / GetLength(MeanGradientShape) >= 1) ? OutputShape[0] : MeanGradientShape[0];
		threads = (GetLength(OutputShape) / GetLength(MeanGradientShape) >= 1) ?
			GetLength(OutputShape) / OutputShape[0] : GetLength(MeanGradientShape) / MeanGradientShape[0];
		DLDMeanGpu << <numbers_of_block, numbers_of_thread >> >
			(Input, Gradient, Mean, Variance, Temp, OutputShape[0], OutputShape[1], OutputShape[2], OutputShape[3],
				MeanGradientShape[0], MeanGradientShape[1], MeanGradientShape[2], MeanGradientShape[3],
				ShapeZ[0], ShapeZ[1], ShapeZ[2], ShapeZ[3], Scale, threads);
		cudaDeviceSynchronize();
		double* Compressed;
		shape CompressedShape = OutputShape;
		CompressedShape[0] = 1;
		cudaMallocManaged(&Compressed, sizeof(double) * GetLength(CompressedShape));
		MatrixOperator::CompressVertically(Temp, OutputShape, Compressed, CompressedShape);
		MatrixOperator::CompressHorizontally(Compressed, CompressedShape,MeanGradient, MeanGradientShape);
		cudaFree(Temp);
		cudaFree(Compressed);
	}
	else {
		DLDMeanCpu();
	}
}




void DLDScaleCpu() {}

__global__ void DLDScaleGpu(double* InputHat, double* Gradient, double* Scale, double* ScaleGradient,
	size_t X1, size_t X2, size_t X3, size_t X4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t Z1, size_t Z2, size_t Z3, size_t Z4, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX % (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX % (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4) +
			O3 % X3 * X4 +
			O4 % X4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		ScaleGradient[IDX] =Scale[idxY]*Gradient[idxX]*InputHat[idxX];
	}
}

void BatchNormOperator::GetDLDScale(double* InputHat, double* Gradient, shape InputShape, double* Scale, double* ScaleGradient, shape ScaleShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		double* Temp;
		shape ShapeZ =InputShape;
		cudaMallocManaged(&Temp, sizeof(double) * GetLength(InputShape));
		numbers_of_block = (GetLength(InputShape) / GetLength(ScaleShape) >= 1) ? InputShape[0] : ScaleShape[0];
		threads = (GetLength(InputShape) / GetLength(ScaleShape) >= 1) ?
			GetLength(InputShape) / InputShape[0] : GetLength(ScaleShape) / ScaleShape[0];
		DLDScaleGpu << <numbers_of_block, numbers_of_thread >> >
			(InputHat,Gradient,Scale, Temp,InputShape[0], InputShape[1], InputShape[2], InputShape[3],
				ScaleShape[0], ScaleShape[1], ScaleShape[2], ScaleShape[3],
				InputShape[0], InputShape[1], InputShape[2], InputShape[3], threads);
		cudaDeviceSynchronize();
		double* Compressed;
		shape CompressedShape = InputShape;
		CompressedShape[0] = 1;
		cudaMallocManaged(&Compressed, sizeof(double) * GetLength(CompressedShape));
		MatrixOperator::CompressVertically(Temp,InputShape, Compressed, CompressedShape);
		MatrixOperator::CompressHorizontally(Compressed, CompressedShape, ScaleGradient,ScaleShape);
		cudaFree(Temp);
		cudaFree(Compressed);
	}
	else {
		DLDScaleCpu();
	}
}





void DLDOffsetCpu() {}

__global__ void DLDOffsetGpu(double* Gradient, double* Scale, double* ScaleGradient,
	size_t X1, size_t X2, size_t X3, size_t X4,
	size_t Y1, size_t Y2, size_t Y3, size_t Y4,
	size_t Z1, size_t Z2, size_t Z3, size_t Z4, size_t threads) {
	size_t idx = threadIdx.x;
	size_t row = blockIdx.x;
	size_t stride = blockDim.x;
	size_t IDX;
	for (int i = idx; i < threads; i = i + stride) {
		IDX = i + blockIdx.x * threads;
		size_t O1 = IDX / (Z2 * Z3 * Z4);
		size_t O2 = (IDX % (Z2 * Z3 * Z4)) / (Z3 * Z4);
		size_t O3 = (IDX % (Z3 * Z4) / Z4);
		size_t O4 = IDX % Z4;

		size_t idxX = O1 % X1 * (X2 * X3 * X4) +
			O2 % X2 * (X3 * X4) +
			O3 % X3 * X4 +
			O4 % X4;
		size_t idxY = O1 % Y1 * (Y2 * Y3 * Y4) +
			O2 % Y2 * (Y3 * Y4) +
			O3 % Y3 * Y4 +
			O4 % Y4;
		ScaleGradient[IDX] = Scale[idxY] * Gradient[idxX];
	}
}

void BatchNormOperator::GetDLDOffset(double* Gradient, shape InputShape, double* Scale, double* OffsetGradient, shape OffsetShape)
{
	cudaError_t status = cudaSetDevice(0);
	if (status == cudaSuccess) {
		size_t numbers_of_block;
		size_t numbers_of_thread = NUMBERS_OF_THREADS;
		size_t threads;
		double* Temp;
		shape ShapeZ = InputShape;
		cudaMallocManaged(&Temp, sizeof(double) * GetLength(InputShape));
		numbers_of_block = (GetLength(InputShape) / GetLength(OffsetShape) >= 1) ? InputShape[0] : OffsetShape[0];
		threads = (GetLength(InputShape) / GetLength(OffsetShape) >= 1) ?
			GetLength(InputShape) / InputShape[0] : GetLength(OffsetShape) / OffsetShape[0];
		DLDOffsetGpu << <numbers_of_block, numbers_of_thread >> >
			( Gradient, Scale,Temp, InputShape[0], InputShape[1], InputShape[2], InputShape[3],
				OffsetShape[0],OffsetShape[1],OffsetShape[2], OffsetShape[3],
				InputShape[0], InputShape[1], InputShape[2], InputShape[3], threads);
		cudaDeviceSynchronize();
		double* Compressed;
		shape CompressedShape = InputShape;
		CompressedShape[0] = 1;
		cudaMallocManaged(&Compressed, sizeof(double) * GetLength(CompressedShape));
		MatrixOperator::CompressVertically(Temp, InputShape, Compressed, CompressedShape);
		MatrixOperator::CompressHorizontally(Compressed, CompressedShape, OffsetGradient, OffsetShape);
		cudaFree(Temp);
		cudaFree(Compressed);
	}
	else {
		DLDOffsetCpu();
	}
}
