#pragma once


typedef std::vector<size_t> shape;
namespace MatrixOperator {
	void Convolution(double* Input, shape InputShape, double* Kernel, shape KernelShape, double* Output, shape OutputShape, size_t stride);
	void Multiply2D(double* X, shape ShapeX, double* Y, shape ShapeY, double* Z, shape ShapeZ);
	void SetZero(double* X, shape ShapeX);
	void Transpose(double* X, shape & ShapeX, double* X_T, shape & ShapeX_T);
	void SetValue(double* X, shape Shape,double Start);
	void SetNumber(double* X, shape Shape, double Value);
	void PrintValue(double* X, shape Shape);
	void MatrixAdd(double* X, shape & ShapeX , double* Y,shape & ShapeY, double* Z, shape & ShapeZ);
	void MatrixSub(double* X, shape & ShapeX, double* Y, shape & ShapeY, double* Z, shape & ShapeZ);
	void Copy(double* FormData, shape Shape, double* TargetData);
	void MultiplyNumber(double* FormData,shape Shape, double* Target,double Number);
	void CompressVertically(double* FormData,  shape FormShape,double* Target,shape TargetShape);/*M*N变为1*N*/
	void CompressHorizontally(double* FormData, shape FormShape, double* Target, shape TargetShape);/*M*N变为M*1*/
	void MatrixSum(double* Data, shape Shape,double& Output);
	void MatrixSquare(double* FormData, shape Shape, double* Target);
	void MatrixNegative(double* FormData, shape Shape, double* TargetData);
	void MatrixExp(double* FormData, shape Shape, double* TargetData);
	void MatrixLog(double* FormData, shape Shape, double* TargetData);
	void MatrixDivision(double* X, shape& ShapeX, double* Y, shape& ShapeY, double* Z, shape& ShapeZ);
	void MatrixMultiply(double* X, shape& ShapeX, double* Y, shape& ShapeY, double* Z, shape& ShapeZ);
	void Multiply3D(double* X, shape ShapeX, double* Y, shape ShapeY, double* Z, shape ShapeZ);
	void MatrixGetLabel(double* FormData, shape FormShape, double* Target, shape TargetShape);
	void Rotation180(double* Former, shape FormerShape, double* Target, shape TargetShape);
}


namespace ReLUOperator {
	void ReLUForward(double* Input,shape Shape,double* Output);
	void ReLUBackward(double* Input,double* Gradient,shape Shape,double* ReLUGradient);
}

namespace ConvOperator {
	void PaddingData(double* Input, shape ShapeInput, double* Output, shape ShapeOut,size_t PaddingHorizontally,size_t PaddingVertically);
	void ConvDLDK(double* Output, shape OutputShape, double* Input, shape InputShape, double* KernelTemp, shape KernelShape, size_t stride);
	void RecoveryPadding(double* PaddedData, shape PaddedShape, double* RecoveryData, shape RecoverShape, size_t PaddingHorizontally, size_t PaddingVertically);
}


namespace PoolingOperator{
	void MaxPoolingConv(double* Input, shape InputShape, double* Kernel, shape KernelShape,
		double* Output, shape OutputShape, size_t Stride);
	void MaxPoolingBackward(double* Input, shape InputShape,double* Output,
		double* Gradient, shape GradientShape,double* MaxPoolingGradient,
		size_t KernelWidth,size_t KernelHeight,size_t Stride);
	void AveragePoolingBackward(double* Input, shape InputShape, double* Output,
		double* Gradient, shape GradientShape, double* MaxPoolingGradient,
		size_t KernelWidth, size_t KernelHeight, size_t Stride);
}

namespace SigmoidOperator {
	void SigmoidForward(double* Input, shape Shape, double* Output);
	void SigmoidBackward(double* Output, double* Gradient, shape Shape, double* ReLUGradient);
}


namespace SoftMaxOperator {
	void SoftMaxForward(double* Input, shape Shape, double* Output);
	void JacobianMatrix(double*Output,shape OutputShape,double* Jacobian, shape JacobianShape);
}



namespace CrossEntropyOperator {
	void OneHot(double* Input, shape InputShape, double* OneHot, shape OneHotShape);
}


void Compare(double* Label, double* Target, size_t size);
namespace WeightInit {
	void Xavier(double* Weight, shape Shape);
	void Kaiming(double* Weight, shape Shape);
	void SetOne(double* Weight, shape Shape);
};

namespace DropOutOperator {
	void DropOutForward(double* Input, shape Shape, double* Output,double Rate);
	void DropOutBackward(double* Input,double* Output, double* Gradient, shape Shape, double* DropOutGradient,double Rate);
}


namespace L1Operator {
	void L1Backward(double* Weight, shape Shape, double* NormBP, double Lambda);
}


namespace BatchNormOperator {
	void CalculateMean(double* Input,shape InputShape,double* Mean,shape MeanShape);
	void CalculateVariance(double* Input, shape InputShape,double* Mean,shape MeanShape,double* Variance, shape VarianceShape);
	void CalculateMoving(double* Moving, shape MovingShape, double* NewValue, shape NewValueShape,double* NewMoving,double Momentum);
	void CalculateValueHat(double* Value, shape ValueShape, double* Mean, shape MeanShape, double* Variance, shape VarianceShape,double* ValueHat);
	void NormalizeBackward(double* Input, double* InputHat, double* Output, double* Gradient,double* BatchGradient,shape Inputshape,
		double* Mean, double* Variance, double* Scale, double* Offset, double* ScaleGradient, 
		double* OffsetGradient, shape MeanShape);
	void GetDLDVariance(double* Input,  shape OutputShape, double* Gradient, double* Mean,
		double* Variance, double* MeanGradient, shape MeanGradientShape, double* Scale);
	void GetDLDMean(double* Input, shape OutputShape, double* Gradient, double* Mean,
		double* Variance, double* MeanGradient, shape  MeanGradientShape, double* Scale);
	void GetDLDScale(double* InputHat,double* Gradient,shape InputShape,double* Scale,double* ScaleGradient,shape ScaleShape);
	void GetDLDOffset( double* Gradient, shape InputShape , double* Scale,double* OffsetGradient, shape OffsetShape);
}

size_t GetMax(double* Data, size_t size);