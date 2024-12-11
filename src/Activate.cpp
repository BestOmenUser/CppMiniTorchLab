#include"Activate.h"
extern "C"  class Model;
ActivateLink::ActivateLink()
{
	head = NULL;
	Tail = NULL;
}

ActivateLink::~ActivateLink()
{
	/*
	Node* temp= head;
	while (temp != NULL) {
		Node* del = temp;
		//delete (*del).func;
		delete del;
		temp = (*temp).next;
	}*/
}


size_t AccuracyNumber = 0;
size_t RightPredict = 0;
size_t LossNumber = 0;
double TotalLoss = 0;

void ShowEvaluate()
{
	if (AccuracyNumber != 0) {
		std::cout << "Total Average Accuracy Rate is:" << (double)RightPredict / (double)AccuracyNumber << std::endl;
		RightPredict = 0;
		AccuracyNumber = 0;
	}
	if (LossNumber != 0) {
		std::cout << "Total Average Loss is:" << (double)TotalLoss / (double)LossNumber << std::endl;
		TotalLoss = 0;
		LossNumber = 0;
	}
}

void Activate::ReLU::SuitData(DataBlock& Former) {
	this->Input = Former;
	this->OutShape = Former.shape;
	this->Do_Di_Shape = Former.shape;
}

void Activate::ReLU::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

void Activate::ReLU::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}




shape Activate::ReLU::ComputeGraph(shape Shape)
{
	std::cout << "ReLU output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < Shape.size(); size++) {
		std::cout << Shape[size];
		if (size != Shape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return Shape;
}


DataBlock Activate::ReLU::Forward(DataBlock &Data, bool train) {
	this->SuitData(Data);

	if (!Status_Init) {
		this->Init();
	}
	this->SetZero();
	
	ReLUOperator::ReLUForward(this->Input.Data, this->Input.shape, this->Output);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Activate::ReLU::BackPropagation(DataBlock& Gradient)
{
	ReLUOperator::ReLUBackward(this->Input.Data,Gradient.Data, Gradient.shape, this->Do_Di);
	DataBlock Dx;
	Dx.Init(Do_Di,	Do_Di_Shape);
	return Dx;
}




void Activate::ReLU::SetZero()
{
}


Activate::ReLU::~ReLU() 
{
}


Layer::Linear::Linear(size_t inputSize, size_t outputSize)
{
	cudaMallocManaged(&Weight, sizeof(double)* inputSize * outputSize);
	cudaMallocManaged(&WeightTemp, sizeof(double) * inputSize * outputSize);
	cudaMallocManaged(&Bias, sizeof(double) * outputSize);
	cudaMallocManaged(&BiasTemp, sizeof(double) * outputSize);
	WeightShape = {inputSize,1,1,outputSize };
	BiasShape = {1,1,1,outputSize };
}


void Layer::Linear::SetWeight(SetWeightPointer WeightInitWay) {
	WeightInitWay(this->Weight, this->WeightShape);
	MatrixOperator::SetZero(this->Bias,this->BiasShape);
}

void Layer::Linear::SuitData(DataBlock& Former) {
	this->Input = Former;
	this->OutShape = Former.shape;
	this->OutShape[3] = this->WeightShape[3];
	this->Do_Di_Shape = Former.shape;
	this->Do_Di_Shape[3] = this->WeightShape[0];
}

void Layer::Linear::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}



shape Layer::Linear::ComputeGraph(shape Shape)
{
	if (Shape[Shape.size() - 1] != this->WeightShape[0]) {
		std::cout << "Shape can not match" << std::endl;
		exit(0);
	}
	std::cout << "Linear output shape:    MiniTorch.Size([" << 1<< "," 
		<< this->WeightShape[3] <<"])" << std::endl;
	return { 1, this->WeightShape[3] };
}

void Layer::Linear::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}

void Layer::Linear::SetNormNumber(NormNumberBase* Norm)
{
	this->LayerNormNumber = Norm;
}

DataBlock Layer::Linear::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
		Init();
	}
	this->SetZero();
	MatrixOperator::Multiply2D(this->Input.Data, this->Input.shape,
		this->Weight, this->WeightShape, this->Output, OutShape);
	MatrixOperator::MatrixAdd(this->Output, this->OutShape,
		this->Bias, this->BiasShape, this->Output, this->OutShape);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Layer::Linear::BackPropagation(DataBlock& Gradient)
{
	double* WeightTranspose;
	double* InputTranspose;
	shape InputTransposeShape;
	shape WeightTransposeShape;
	size_t WeightSize = GetLength(this->WeightShape);
	size_t InputSize = GetLength(this->Input.shape);
	cudaMallocManaged(&WeightTranspose, WeightSize * sizeof(double));
	cudaMallocManaged(&InputTranspose,InputSize * sizeof(double)); 
	MatrixOperator::Transpose(this->Input.Data,this->Input.shape,InputTranspose,InputTransposeShape);
	MatrixOperator::Multiply2D(InputTranspose,InputTransposeShape,
		Gradient.Data, Gradient.shape, this->WeightTemp, this->WeightShape);

	if (this->LayerNormNumber) {
		double* NormNumberGradient;
		cudaMallocManaged(&NormNumberGradient,WeightSize*sizeof(double));
		(*this->LayerNormNumber).NormBackPropagation(this->Weight, this->WeightShape, NormNumberGradient);
		MatrixOperator::MatrixAdd(this->WeightTemp,this->WeightShape,
			NormNumberGradient,this->WeightShape, this->WeightTemp,this->WeightShape);
		cudaFree(NormNumberGradient);
	}

	MatrixOperator::CompressVertically(Gradient.Data, Gradient.shape,this->BiasTemp,this->BiasShape);

	MatrixOperator::Transpose(this->Weight, this->WeightShape, WeightTranspose, WeightTransposeShape);
	MatrixOperator::Multiply2D(Gradient.Data, Gradient.shape,
		WeightTranspose, WeightTransposeShape, this->Do_Di, this->Do_Di_Shape);
	cudaFree(WeightTranspose);
	cudaFree(InputTranspose);
	DataBlock Dx;
	Dx.Init(this->Do_Di, this->Do_Di_Shape);
	return Dx;
}

void Layer::Linear::SetZero()
{
}


void Layer::Linear::Update(double learningRate)
{
	MatrixOperator::MultiplyNumber(this->WeightTemp, this->WeightShape, this->WeightTemp, learningRate);
	MatrixOperator::MultiplyNumber(this->BiasTemp, this->BiasShape, this->BiasTemp, learningRate);
	MatrixOperator::MatrixSub(this->Weight, this->WeightShape,
		this->WeightTemp, this->WeightShape, this->Weight, this->WeightShape);
	MatrixOperator::MatrixSub(this->Bias, this->BiasShape,
		this->BiasTemp, this->BiasShape, this->Bias, this->BiasShape);
}


Layer::Linear::~Linear()
{
	cudaFree(this->Weight);
	cudaFree(this->WeightTemp);
	cudaFree(this->Bias);
	cudaFree(this->BiasTemp);
}


void Loss::MSE::SuitData(DataBlock& Former) {
	this->Input = Former;
	this->OutShape = std::vector<size_t>{ 1,1,1,1 };
	this->Do_Di_Shape = Former.shape;
}

void Loss::MSE::Init(){
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

void Loss::MSE::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}


DataBlock Loss::MSE::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
		this->Init();
	}
	this->SetZero();
	double* OutputSubInput;
	shape OutputSubInputShape = this->Input.shape;
	size_t size = GetLength(OutputSubInputShape);
	cudaMallocManaged(&OutputSubInput, size*sizeof(double));
	MatrixOperator::MatrixSub(this->Input.Data, this->Input.shape, this->Label.Data, this->Label.shape,
		OutputSubInput, OutputSubInputShape);
	MatrixOperator::MatrixSquare(OutputSubInput, OutputSubInputShape, OutputSubInput);
	MatrixOperator::MatrixSum(OutputSubInput, OutputSubInputShape, *this->Output);
	LossNumber += Input.shape[0];
	TotalLoss += *Output/2;
	*this->Output = *this->Output / (2 * this->Input.shape[0]);
	if (LossNumber < Label.shape[0]&&!Train) {
		Compare(Label.Data,Data.Data,Label.shape[0]);
	}
	std::cout << "Average Loss:" << *Output << std::endl;
	cudaFree(OutputSubInput);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Loss::MSE::BackPropagation(DataBlock& Gradient)
{
	MatrixOperator::MatrixSub(this->Input.Data, this->Input.shape, this->Label.Data, this->Label.shape,
		this->Do_Di, this->Do_Di_Shape);
	MatrixOperator::MultiplyNumber(this->Do_Di, this->Do_Di_Shape, this->Do_Di, 1.0/this->Input.shape[0]);
	DataBlock Dx;
	Dx.Init(this->Do_Di,this->Do_Di_Shape);
	return Dx;
}

void Loss::MSE::SetNormNumber(NormNumberBase* Norm) {
	this->LayerNormNumber = Norm;
}


void Loss::MSE::SetZero()
{
}

shape Loss::MSE::ComputeGraph(shape Shape)
{
	return Shape;
}

void Loss::MSE::SetLabel(DataBlock& Label)
{
	this->Label = Label;
}

Loss::MSE::~MSE()
{
}



void Layer::Flatten::SuitData(DataBlock& Former) {
	this->Input = Former;
	this->OutShape = { 1,1,1,1 };
	this->OutShape[0] = Former.shape[0];
	this->OutShape[3] = GetLength(Former.shape)/Former.shape[0];
	this->Do_Di_Shape = Former.shape;
}

void Layer::Flatten::Init() {
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

void Layer::Flatten::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}



DataBlock Layer::Flatten::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
			this->Init();
			Status_Init = true;
		}
	MatrixOperator::Copy(Data.Data, Data.shape, Output);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Layer::Flatten::BackPropagation(DataBlock& Data)
{
	MatrixOperator::Copy(Data.Data, Data.shape, Do_Di);
	DataBlock Dx;
	Dx.Init(this->Do_Di,this->Do_Di_Shape);
	return Dx;
}


void Layer::Flatten::SetZero()
{
}

shape Layer::Flatten::ComputeGraph(shape Shape)
{
	std::cout << "Flatten output shape:    MiniTorch.Size([" << 1 << ","
		<< GetLength(Shape) << "])" << std::endl;
	return { 1, GetLength(Shape) };
}

Layer::Flatten::~Flatten()
{

}



void Activate::SoftMax::SuitData(DataBlock& Former) {
	this->Input = Former;
	this->OutShape = Former.shape;
	this->Do_Di_Shape = Former.shape;
}

void Activate::SoftMax::SetZero()
{
}

void Activate::SoftMax::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

void Activate::SoftMax::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}


shape Activate::SoftMax::ComputeGraph(shape Shape)
{
	std::cout << "SoftMax output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < Shape.size(); size++) {
		std::cout << Shape[size];
		if (size != Shape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return Shape;
}

DataBlock Activate::SoftMax::Forward(DataBlock& Data, bool Train)
{
	this->SuitData(Data);

	if (!Status_Init) {
		this->Init();
	}

	this->SetZero();
	SoftMaxOperator::SoftMaxForward(this->Input.Data, this->Input.shape, this->Output);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Activate::SoftMax::BackPropagation(DataBlock& Gradient)
{
	double* Jacobian;
	shape JacobianShape = this->Input.shape;
	JacobianShape[2] = JacobianShape[3];
	cudaMallocManaged(&Jacobian, GetLength(JacobianShape) * sizeof(double));
	SoftMaxOperator::JacobianMatrix(this->Output,this->OutShape,Jacobian, JacobianShape);
	MatrixOperator::Multiply3D(Gradient.Data, Gradient.shape, Jacobian, JacobianShape, Do_Di, Do_Di_Shape);
	DataBlock Dx;
	Dx.Init(Do_Di, Do_Di_Shape);
	cudaFree(Jacobian);
	return Dx;
}


void Loss:: CrossEntropyLoss::SetLabel(DataBlock& Label)
{
	this->Label = Label;
}

void Loss::CrossEntropyLoss::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

shape Loss::CrossEntropyLoss::ComputeGraph(shape Shape)
{
	return shape();
}

void Loss::CrossEntropyLoss::DestroyInit()
{
	if (LabelInit) {
		cudaFree(this->OneHot.Data);
		LabelInit = false;
	}
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}

void Loss::CrossEntropyLoss::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = std::vector<size_t>{ 1,1,1,1 };
	this->Do_Di_Shape = Former.shape;
}

void Loss::CrossEntropyLoss::SetZero()
{
}

Loss::CrossEntropyLoss::~CrossEntropyLoss()
{
}

void Loss::CrossEntropyLoss::MakeOneHot(size_t KindNums)
{
	size_t Numbers = KindNums;
	this->OneHot.shape = this->Label.shape;
	this->OneHot.shape[3] = Numbers;
	if (!LabelInit) {
		cudaMallocManaged(&this->OneHot.Data, sizeof(double) * GetLength(this->OneHot.shape));
		LabelInit = true;
	}
	CrossEntropyOperator::OneHot(this->Label.Data, this->Label.shape, this->OneHot.Data, this->OneHot.shape);
}

void Loss::CrossEntropyLoss::GetAccuracy(DataBlock Label, DataBlock Predicted,bool Train)
{
	shape PredictedLabelShape = Label.shape;
	double* PredictedLabel;
	double accuracy = 0.0;
	cudaMallocManaged(&PredictedLabel, sizeof(double) * GetLength(PredictedLabelShape));
	MatrixOperator::MatrixGetLabel(Predicted.Data, Predicted.shape, PredictedLabel, PredictedLabelShape);
	for (auto i = 0; i < Label.shape[0]; i++) {
		if (PredictedLabel[i] == Label.Data[i]) {
			accuracy++;
		}
	}
	if (!Train) {
		if (AccuracyNumber < Label.shape[0]) {
			Compare(Label.Data, PredictedLabel, Label.shape[0]);
		}
	}
	AccuracyNumber += Label.shape[0];
	RightPredict += accuracy;
	std::cout << "Average Accuracy is:" << accuracy / Label.shape[0] << std::endl;
	cudaFree(PredictedLabel);
}


DataBlock Loss::CrossEntropyLoss::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
		this->Init();
	}
	MakeOneHot(Data.shape[3]);
	this->SetZero();
	double* InputLog;
	cudaMallocManaged(&InputLog, GetLength(this->Input.shape) * sizeof(double));
	MatrixOperator::MatrixLog(this->Input.Data, this->Input.shape, InputLog);
	MatrixOperator::MatrixMultiply(this->OneHot.Data, this->OneHot.shape,
		InputLog, this->Input.shape, InputLog, this->Input.shape);
	MatrixOperator::MatrixSum(InputLog, this->Input.shape, *this->Output);
	LossNumber += Input.shape[0];
	TotalLoss += -*Output;

	*this->Output = -*this->Output / this->Input.shape[0];
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	GetAccuracy(Label, Data,Train);
	std::cout << "Average Loss:" << *Output<< std::endl;
	cudaFree(InputLog);
	return next_data;
}

DataBlock Loss::CrossEntropyLoss::BackPropagation(DataBlock& Gradient)
{
	shape OneSize = { 1,1,1,1 };
	double* One;
	shape BatchSize = { 1,1,1,1 };
	double* Batch;
	cudaMallocManaged(&One, GetLength(OneSize));
	cudaMallocManaged(&Batch, GetLength(BatchSize));
	*One = 1.0;
	*Batch = this->Input.shape[0];
	MatrixOperator::MatrixDivision(One, OneSize, this->Input.Data, this->Input.shape, this->Do_Di, this->Do_Di_Shape);
	MatrixOperator::MatrixMultiply(this->OneHot.Data, this->OneHot.shape, this->Do_Di, this->Do_Di_Shape, this->Do_Di, this->Do_Di_Shape);
	MatrixOperator::MatrixNegative(this->Do_Di, this->Do_Di_Shape,this->Do_Di);
	MatrixOperator::MatrixMultiply(Batch, BatchSize, this->Do_Di, this->Do_Di_Shape, this->Do_Di, this->Do_Di_Shape);
	DataBlock Dx;
	Dx.Init(this->Do_Di, this->Do_Di_Shape);
	cudaFree(One);
	cudaFree(Batch);
	return Dx;
}


DataBlock Activate::Sigmoid::Forward(DataBlock& Data, bool Train)
{
	this->SuitData(Data);

	if (!Status_Init) {
		this->Init();
	}
	this->SetZero();

	SigmoidOperator::SigmoidForward(this->Input.Data, this->Input.shape, this->Output);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Activate::Sigmoid::BackPropagation(DataBlock& Gradient)
{
	SigmoidOperator::SigmoidBackward(this->Output, Gradient.Data, Gradient.shape, this->Do_Di);
	DataBlock Dx;
	Dx.Init(Do_Di, Do_Di_Shape);
	return Dx;
}

void Activate::Sigmoid::SetZero()
{
}

void Activate::Sigmoid::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

shape Activate::Sigmoid::ComputeGraph(shape Shape)
{
	std::cout << "Sigmoid output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < Shape.size(); size++) {
		std::cout << Shape[size];
		if (size != Shape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return Shape;
}

void Activate::Sigmoid::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}

void Activate::Sigmoid::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = Former.shape;
	this->Do_Di_Shape = Former.shape;
}

Activate::Sigmoid::~Sigmoid()
{
}

Layer::Conv2d::Conv2d(size_t inputChanell, size_t outputChanel, size_t kernelSizeWidth,size_t kernelSizeHeight,
	size_t stride, size_t HorizontallyPadding , size_t VerticallyPadding )
{
	KernelShape = { outputChanel,inputChanell,kernelSizeWidth,kernelSizeHeight };
	BiasShape = { 1,outputChanel,1,1 };
	Stride = stride;
	this->VerticallyPadding = VerticallyPadding;
	this->HorizontallyPadding = HorizontallyPadding;
	cudaMallocManaged(&Kernel, sizeof(double) *GetLength(KernelShape));
	cudaMallocManaged(&KernelTemp, sizeof(double) *GetLength(KernelShape));
	cudaMallocManaged(&Bias, sizeof(double) *GetLength(BiasShape));
	cudaMallocManaged(&BiasTemp, sizeof(double) * GetLength(BiasShape));
}


DataBlock Layer::Conv2d::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
		Init();
	}
	if (VerticallyPadding != 0||HorizontallyPadding!=0) {
		MakePadding();
	}
	this->SetZero();
	MatrixOperator::Convolution((VerticallyPadding != 0 || HorizontallyPadding != 0 )? this->ProcessedData.Data : this->Input.Data,
		(VerticallyPadding != 0 || HorizontallyPadding != 0 )? this->ProcessedData.shape: this->Input.shape ,
		this->Kernel,this->KernelShape,this->Output,this->OutShape,this->Stride);
	MatrixOperator::MatrixAdd(this->Output, this->OutShape, this->Bias, this->BiasShape,this->Output,this->OutShape);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Layer::Conv2d::BackPropagation(DataBlock& Gradient)
{
	double* KernelRotation;
	shape RotatedShape = this->KernelShape;
	RotatedShape[0] = this->KernelShape[1];
	RotatedShape[1] = this->KernelShape[0];
	cudaMallocManaged(&KernelRotation, sizeof(double) * GetLength(RotatedShape));
	MatrixOperator::Rotation180(this->Kernel, this->KernelShape, KernelRotation, RotatedShape);
	double* PaddedGradient;
	shape PaddedShape = Gradient.shape;
	PaddedShape[2] += 2 * (this->KernelShape[2] - 1);
	PaddedShape[3] += 2 * (this->KernelShape[3] - 1);
	cudaMallocManaged(&PaddedGradient, sizeof(double) * GetLength(PaddedShape));
	cudaMemset(PaddedGradient, 0, sizeof(double) * GetLength(PaddedShape));
	ConvOperator::PaddingData(Gradient.Data, Gradient.shape,
		PaddedGradient, PaddedShape, this->KernelShape[2] - 1, this->KernelShape[3] - 1);

	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		double* Do_Di_Padded;
		cudaMallocManaged(&Do_Di_Padded, sizeof(double) * GetLength(this->ProcessedData.shape));
		cudaMemset(Do_Di_Padded, 0, sizeof(double) * GetLength(this->ProcessedData.shape));
		MatrixOperator::Convolution(PaddedGradient, PaddedShape, KernelRotation, RotatedShape,
			Do_Di_Padded, this->ProcessedData.shape, this->Stride);
		ConvOperator::RecoveryPadding(Do_Di_Padded, this->ProcessedData.shape,
			this->Do_Di, this->Do_Di_Shape, this->HorizontallyPadding, this->VerticallyPadding);
		cudaFree(Do_Di_Padded);
	}
	else {
		MatrixOperator::Convolution(PaddedGradient, PaddedShape, KernelRotation, RotatedShape,
			Do_Di,this->Do_Di_Shape, this->Stride);
	}
	double* BiasCompressed;
	shape BiasCompressedShape = this->BiasShape;
	BiasCompressedShape[0] = Gradient.shape[0];
	cudaMallocManaged(&BiasCompressed, sizeof(double) * GetLength(BiasCompressedShape));
	MatrixOperator::CompressHorizontally(Gradient.Data, Gradient.shape, BiasCompressed, BiasCompressedShape);
	MatrixOperator::CompressVertically(BiasCompressed, BiasCompressedShape, this->BiasTemp, this->BiasShape);

	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		ConvOperator::ConvDLDK(Gradient.Data, Gradient.shape, this->ProcessedData.Data, this->ProcessedData.shape,
			this->KernelTemp, this->KernelShape, this->Stride);
	}
	else {
		ConvOperator::ConvDLDK(Gradient.Data, Gradient.shape, this->Input.Data, this->Input.shape,
			this->KernelTemp, this->KernelShape, this->Stride);
	}

	if (this->LayerNormNumber) {
		double* NormNumberGradient;
		cudaMallocManaged(&NormNumberGradient,GetLength(this->KernelShape)* sizeof(double));
		(*this->LayerNormNumber).NormBackPropagation(this->Kernel, this->KernelShape, NormNumberGradient);
		MatrixOperator::MatrixAdd(this->KernelTemp, this->KernelShape,
			NormNumberGradient, this->KernelShape, this->KernelTemp, this->KernelShape);
		cudaFree(NormNumberGradient);
	}
	DataBlock Dx;
	Dx.Init(this->Do_Di, this->Do_Di_Shape);
	cudaFree(BiasCompressed);
	cudaFree(KernelRotation);
	cudaFree(PaddedGradient);
	return Dx;
}


void Layer::Conv2d::SetZero()
{
}

void Layer::Conv2d::SetWeight(SetWeightPointer WeightInitWay)
{
	WeightInitWay(this->Kernel, this->KernelShape);
	MatrixOperator::SetZero(this->Bias, this->BiasShape);
}

void Layer::Conv2d::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		cudaMallocManaged(&this->ProcessedData.Data, sizeof(double) * GetLength(ProcessedData.shape));
		cudaMemset(ProcessedData.Data, 0, sizeof(double) * GetLength(ProcessedData.shape));
	}
	this->Status_Init = true;
}

shape Layer::Conv2d::ComputeGraph(shape Shape)
{
	shape NextShape=Shape;
	NextShape[1] = this->KernelShape[0];
	NextShape[2]=(Shape[2] + 2 * HorizontallyPadding - KernelShape[2] ) / Stride+1;
	NextShape[3]=(Shape[3] + 2 *VerticallyPadding- KernelShape[3] ) / Stride+1;
	std::cout << "Conv output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < NextShape.size(); size++) {
		std::cout << NextShape[size];
		if (size != NextShape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return NextShape;
}

void Layer::Conv2d::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		cudaFree(this->ProcessedData.Data);
	}
	this->Status_Init = false;
}

void Layer::Conv2d::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = Former.shape;
	this->OutShape[1] = KernelShape[0];
	this->OutShape[2] = (Former.shape[2] + 2 * HorizontallyPadding-KernelShape[2]) / Stride+1;
	this->OutShape[3] = (Former.shape[3] + 2 * VerticallyPadding- KernelShape[3]) / Stride+1;
	this->Do_Di_Shape = Former.shape;
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
			ProcessedData.shape = Input.shape;
			ProcessedData.shape[2] +=HorizontallyPadding*2;
			ProcessedData.shape[3] +=VerticallyPadding* 2;
	}
}

void Layer::Conv2d::Update(double learningRate)
{
	MatrixOperator::MultiplyNumber(this->KernelTemp, this->KernelShape, this->KernelTemp, learningRate);
	MatrixOperator::MultiplyNumber(this->BiasTemp, this->BiasShape, this->BiasTemp, learningRate);
	MatrixOperator::MatrixSub(this->Kernel, this->KernelShape,
		this->KernelTemp, this->KernelShape, this->Kernel, this->KernelShape);
	MatrixOperator::MatrixSub(this->Bias, this->BiasShape,
		this->BiasTemp, this->BiasShape, this->Bias, this->BiasShape);
}

void Layer::Conv2d::SetNormNumber(NormNumberBase* Norm)
{
	this->LayerNormNumber = Norm;
}

Layer::Conv2d::~Conv2d()
{
	cudaFree(this->Kernel);
	cudaFree(this->KernelTemp);
	cudaFree(this->Bias);
	cudaFree(this->BiasTemp);
}

void Layer::Conv2d::MakePadding()
{
	ConvOperator::PaddingData(this->Input.Data, this->Input.shape,
		this->ProcessedData.Data, this->ProcessedData.shape,this->HorizontallyPadding,this->VerticallyPadding);
}




Layer::MaxPooling::MaxPooling(size_t kernelSizeWidth, size_t kernelSizeHeight, size_t stride, size_t HorizontallyPadding, size_t VerticallyPadding)
{
	KernelShape = { 1,1,kernelSizeWidth,kernelSizeHeight };
	Stride = stride;
	this->VerticallyPadding = VerticallyPadding;
	this->HorizontallyPadding = HorizontallyPadding;
	cudaMallocManaged(&Kernel, sizeof(double) * GetLength(KernelShape));
}

DataBlock Layer::MaxPooling::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
		Init();
	}
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		MakePadding();
	}
	this->SetZero();
	PoolingOperator::MaxPoolingConv ((VerticallyPadding != 0 || HorizontallyPadding != 0) ? this->ProcessedData.Data : this->Input.Data,
		(VerticallyPadding != 0 || HorizontallyPadding != 0) ? this->ProcessedData.shape : this->Input.shape,
		this->Kernel, this->KernelShape, this->Output, this->OutShape, this->Stride);
	
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Layer::MaxPooling::BackPropagation(DataBlock& Gradient)
{
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		double* Do_Di_Padded;
		cudaMallocManaged(&Do_Di_Padded, sizeof(double) * GetLength(this->ProcessedData.shape));
		cudaMemset(Do_Di_Padded, 0, sizeof(double) * GetLength(this->ProcessedData.shape));
		PoolingOperator::MaxPoolingBackward(this->ProcessedData.Data,this->ProcessedData.shape,
			this->Output,Gradient.Data, Gradient.shape,Do_Di_Padded, this->KernelShape[2],this->KernelShape[3], this->Stride);
		ConvOperator::RecoveryPadding(Do_Di_Padded, this->ProcessedData.shape,
			this->Do_Di, this->Do_Di_Shape, this->HorizontallyPadding, this->VerticallyPadding);
		cudaFree(Do_Di_Padded);
	}
	else {
		PoolingOperator::MaxPoolingBackward(this->Input.Data, this->Input.shape,
			this->Output, Gradient.Data, Gradient.shape, this->Do_Di, this->KernelShape[2], this->KernelShape[3], this->Stride);
	}

	DataBlock Dx;
	Dx.Init(this->Do_Di, this->Do_Di_Shape);
	return Dx;
}

void Layer::MaxPooling::SetZero()
{
	cudaMemset(this->Do_Di, 0, GetLength(this->Do_Di_Shape) * sizeof(double));
}

void Layer::MaxPooling::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		cudaMallocManaged(&this->ProcessedData.Data, sizeof(double) * GetLength(ProcessedData.shape));
		cudaMemset(ProcessedData.Data, 0, sizeof(double) * GetLength(ProcessedData.shape));
	}
	this->Status_Init = true;
}

shape Layer::MaxPooling::ComputeGraph(shape Shape)
{
	shape NextShape = Shape;
	NextShape[1] = this->KernelShape[0];
	NextShape[2] = (Shape[2] + 2 * HorizontallyPadding - KernelShape[2]) / Stride + 1;
	NextShape[3] = (Shape[3] + 2 * VerticallyPadding - KernelShape[3]) / Stride + 1;
	std::cout << "MaxPooling output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < NextShape.size(); size++) {
		std::cout << NextShape[size];
		if (size != NextShape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return NextShape;
}

void Layer::MaxPooling::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		cudaFree(this->ProcessedData.Data);
	}
	this->Status_Init = false;
}

void Layer::MaxPooling::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = Former.shape;
	this->OutShape[2] = (Former.shape[2] + 2 * HorizontallyPadding - KernelShape[2]) / Stride + 1;
	this->OutShape[3] = (Former.shape[3] + 2 * VerticallyPadding - KernelShape[3]) / Stride + 1;
	this->Do_Di_Shape = Former.shape;
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		ProcessedData.shape = Input.shape;
		ProcessedData.shape[2] += HorizontallyPadding * 2;
		ProcessedData.shape[3] += VerticallyPadding * 2;
	}
}


Layer::MaxPooling::~MaxPooling()
{
	cudaFree(this->Kernel);
}

void Layer::MaxPooling::MakePadding()
{
	ConvOperator::PaddingData(this->Input.Data, this->Input.shape,
		this->ProcessedData.Data, this->ProcessedData.shape, this->HorizontallyPadding, this->VerticallyPadding);
}


Layer::AvgPooling::AvgPooling(size_t kernelSizeWidth, size_t kernelSizeHeight, size_t stride, size_t HorizontallyPadding, size_t VerticallyPadding)
{
	KernelShape = { 1,1,kernelSizeWidth,kernelSizeHeight };
	Stride = stride;
	this->VerticallyPadding = VerticallyPadding;
	this->HorizontallyPadding = HorizontallyPadding;
	cudaMallocManaged(&Kernel, sizeof(double) * GetLength(KernelShape));
	MatrixOperator::SetNumber(Kernel, KernelShape, 1.0 / GetLength(KernelShape));
}

DataBlock Layer::AvgPooling::Forward(DataBlock& Data, bool Train)
{
	SuitData(Data);
	if (!Status_Init) {
		Init();
	}
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		MakePadding();
	}
	this->SetZero();
	MatrixOperator::Convolution((VerticallyPadding != 0 || HorizontallyPadding != 0) ? this->ProcessedData.Data : this->Input.Data,
		(VerticallyPadding != 0 || HorizontallyPadding != 0) ? this->ProcessedData.shape : this->Input.shape,
		this->Kernel, this->KernelShape, this->Output, this->OutShape, this->Stride);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Layer::AvgPooling::BackPropagation(DataBlock& Gradient)
{
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		double* Do_Di_Padded;
		cudaMallocManaged(&Do_Di_Padded, sizeof(double) * GetLength(this->ProcessedData.shape));
		cudaMemset(Do_Di_Padded, 0, sizeof(double) * GetLength(this->ProcessedData.shape));
		PoolingOperator::AveragePoolingBackward(this->ProcessedData.Data, this->ProcessedData.shape,
			this->Output, Gradient.Data, Gradient.shape, Do_Di_Padded, this->KernelShape[2], this->KernelShape[3], this->Stride);
		ConvOperator::RecoveryPadding(Do_Di_Padded, this->ProcessedData.shape,
			this->Do_Di, this->Do_Di_Shape, this->HorizontallyPadding, this->VerticallyPadding);
		cudaFree(Do_Di_Padded);
	}
	else {
		PoolingOperator::AveragePoolingBackward(this->Input.Data, this->Input.shape,
			this->Output, Gradient.Data, Gradient.shape, this->Do_Di, this->KernelShape[2], this->KernelShape[3], this->Stride);
	}
	DataBlock Dx;
	Dx.Init(this->Do_Di, this->Do_Di_Shape);
	return Dx;
}


void Layer::AvgPooling::SetZero()
{
	cudaMemset(this->Do_Di, 0, GetLength(this->Do_Di_Shape) * sizeof(double));
}

void Layer::AvgPooling::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		cudaMallocManaged(&this->ProcessedData.Data, sizeof(double) * GetLength(ProcessedData.shape));
		cudaMemset(ProcessedData.Data, 0, sizeof(double) * GetLength(ProcessedData.shape));
	}
	this->Status_Init = true;
}

shape Layer::AvgPooling::ComputeGraph(shape Shape)
{
	shape NextShape = Shape;
	NextShape[1] = this->KernelShape[0];
	NextShape[2] = (Shape[2] + 2 * HorizontallyPadding - KernelShape[2]) / Stride + 1;
	NextShape[3] = (Shape[3] + 2 * VerticallyPadding - KernelShape[3]) / Stride + 1;
	std::cout << "AvgPooling output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < NextShape.size(); size++) {
		std::cout << NextShape[size];
		if (size != NextShape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return NextShape;
}

void Layer::AvgPooling::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		cudaFree(this->ProcessedData.Data);
	}
	this->Status_Init = false;
}

void Layer::AvgPooling::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = Former.shape;
	this->OutShape[2] = (Former.shape[2] + 2 * HorizontallyPadding - KernelShape[2]) / Stride + 1;
	this->OutShape[3] = (Former.shape[3] + 2 * VerticallyPadding - KernelShape[3]) / Stride + 1;
	this->Do_Di_Shape = Former.shape;
	if (VerticallyPadding != 0 || HorizontallyPadding != 0) {
		ProcessedData.shape = Input.shape;
		ProcessedData.shape[2] += HorizontallyPadding * 2;
		ProcessedData.shape[3] += VerticallyPadding * 2;
	}
}

Layer::AvgPooling::~AvgPooling()
{
	cudaFree(this->Kernel);
}

void Layer::AvgPooling::MakePadding()
{
	ConvOperator::PaddingData(this->Input.Data, this->Input.shape,
		this->ProcessedData.Data, this->ProcessedData.shape, this->HorizontallyPadding, this->VerticallyPadding);
}

Layer::DropOut::DropOut(double Rate)
{
	this->Rate = Rate;
}

DataBlock Layer::DropOut::Forward(DataBlock& Data, bool Train)
{
	this->SuitData(Data);

	if (!Status_Init) {
		this->Init();
	}
	this->SetZero();
	DropOutOperator::DropOutForward(this->Input.Data, this->Input.shape, this->Output,this->Rate);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Layer::DropOut::BackPropagation(DataBlock& Gradient)
{
	DropOutOperator::DropOutBackward (this->Input.Data,this->Output,Gradient.Data, Gradient.shape, this->Do_Di,this->Rate);
	DataBlock Dx;
	Dx.Init(Do_Di, Do_Di_Shape);
	return Dx;
}

void Layer::DropOut::SetZero()
{

}

void Layer::DropOut::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	this->Status_Init = true;
}

shape Layer::DropOut::ComputeGraph(shape Shape)
{
	std::cout << "DropOut output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < Shape.size(); size++) {
		std::cout << Shape[size];
		if (size != Shape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return Shape;
}

void Layer::DropOut::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	this->Status_Init = false;
}

void Layer::DropOut::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = Former.shape;
	this->Do_Di_Shape = Former.shape;
}

Layer::DropOut::~DropOut()
{
}

Activate::BatchNorm::BatchNorm(double Scale, double Offset, double Momentum)
{
	this->ScaleNumber = Scale;
	this->OffsetNumber = Offset;
	this->Momentum = Momentum;
}

DataBlock Activate::BatchNorm::Forward(DataBlock& Data, bool Train)
{
	this->SuitData(Data);

	if (!Status_Init) {
		this->Init();
	}
	this->SetZero();

	if (!Train) {
		BatchNormOperator::CalculateValueHat(this->Input.Data, this->Input.shape, this->MovingMean, this->MeanShape,
			this->MovingVariance, this->VarianceShape, this->InputHat);
		}
	else {
		BatchNormOperator::CalculateMean(this->Input.Data, this->Input.shape, this->Mean, this->MeanShape);
		double a;
		MatrixOperator::MatrixSum(this->Input.Data, this->Input.shape, a);
		MatrixOperator::MatrixSum(this->Mean, this->MeanShape, a);
		BatchNormOperator::CalculateVariance(this->Input.Data, this->Input.shape,
			this->Mean, this->MeanShape, this->Variance, this->VarianceShape);
		BatchNormOperator::CalculateValueHat(this->Input.Data, this->Input.shape, this->Mean, this->MeanShape,
			this->Variance, this->VarianceShape, this->InputHat);
		BatchNormOperator::CalculateMoving(this->MovingMean, this->MeanShape, this->Mean, this->MeanShape, this->MovingMean, this->Momentum);
		BatchNormOperator::CalculateMoving(this->MovingVariance, this->VarianceShape, this->Variance, this->VarianceShape, this->MovingVariance, this->Momentum);
	}
	
	MatrixOperator::MatrixMultiply(this->InputHat, this->Input.shape, this->Scale, this->MeanShape, this->Output, this->Input.shape);
	MatrixOperator::MatrixAdd(this->Output, this->Input.shape, this->Offset, this->MeanShape, this->Output, this->OutShape);
	DataBlock next_data;
	next_data.Init(this->Output, this->OutShape);
	return next_data;
}

DataBlock Activate::BatchNorm::BackPropagation(DataBlock& Gradient)
{
	BatchNormOperator::NormalizeBackward(this->Input.Data,this->InputHat,this->Output,Gradient.Data,this->Do_Di,this->Input.shape,
		this->Mean,this->Variance,this->Scale,this->Offset,this->ScaleGradient,this->OffsetGradient,MeanShape);
	DataBlock Dx;
	Dx.Init(Do_Di, Do_Di_Shape);
	return Dx;
}

void Activate::BatchNorm::SetZero()
{
}

void Activate::BatchNorm::Update(double learningRate)
{
	MatrixOperator::MultiplyNumber(this->ScaleGradient, this->MeanShape, this->ScaleGradient, learningRate);
	MatrixOperator::MultiplyNumber(this->OffsetGradient, this->MeanShape, this->OffsetGradient, learningRate);
	MatrixOperator::MatrixSub(this->Scale, this->MeanShape,
		this->ScaleGradient, this->MeanShape, this->Scale, this->MeanShape);
	MatrixOperator::MatrixSub(this->Offset, this->MeanShape,
		this->OffsetGradient, this->MeanShape, this->Offset, this->MeanShape);
}

void Activate::BatchNorm::Init()
{
	size_t size = GetLength(this->Do_Di_Shape);
	cudaMallocManaged(&this->Do_Di, size * sizeof(double));
	size = GetLength(this->OutShape);
	cudaMallocManaged(&this->Output, size * sizeof(double));
	cudaMallocManaged(&this->InputHat, size * sizeof(double));

	
	cudaMallocManaged(&this->Mean, sizeof(double) * GetLength(this->MeanShape));
	cudaMallocManaged(&this->Variance, sizeof(double) * GetLength(this->VarianceShape));
	cudaMallocManaged(&this->ScaleGradient, sizeof(double) * GetLength(this->MeanShape));
	cudaMallocManaged(&this->OffsetGradient, sizeof(double) * GetLength(this->MeanShape));
	if (!InitParams) {
		cudaMallocManaged(&this->Scale, sizeof(double) * GetLength(this->MeanShape));
		cudaMallocManaged(&this->Offset, sizeof(double) * GetLength(this->MeanShape));
		MatrixOperator::SetNumber(this->Scale, this->MeanShape, this->ScaleNumber);
		MatrixOperator::SetNumber(this->Offset, this->MeanShape, this->OffsetNumber);
		cudaMallocManaged(&this->MovingMean,  sizeof(double) * GetLength(this->MeanShape));
		cudaMallocManaged(&this->MovingVariance, sizeof(double) * GetLength(this->MeanShape));
		memset(this->MovingMean, 0, sizeof(double) * GetLength(this->MeanShape));
		memset(this->MovingVariance, 0, sizeof(double) * GetLength(this->MeanShape));
		InitParams = true;
	}
	this->Status_Init = true;
}

shape Activate::BatchNorm::ComputeGraph(shape Shape)
{
	std::cout << "BatchNorm output shape:    MiniTorch.Size([";
	for (size_t size = 0; size < Shape.size(); size++) {
		std::cout << Shape[size];
		if (size != Shape.size() - 1) {
			std::cout << ",";
		}
	}
	std::cout << "])" << std::endl;
	return Shape;
}

void Activate::BatchNorm::DestroyInit()
{
	cudaFree(this->Do_Di);
	cudaFree(this->Output);
	cudaFree(this->Mean);
	cudaFree(this->Variance);
	cudaFree(this->InputHat);
	cudaFree(this->ScaleGradient);
	cudaFree(this->OffsetGradient);
	this->Status_Init = false;
}

void Activate::BatchNorm::SuitData(DataBlock& Former)
{
	this->Input = Former;
	this->OutShape = Former.shape;
	this->Do_Di_Shape = Former.shape;
	if (Input.shape[1] == 1 && Input.shape[2] == 1) {
		this->Type = "FC";
	}
	else {
		this->Type = "CONV";
	}
	this->MeanShape = { 1,1,1,1 };
	this->VarianceShape = { 1,1,1,1 };
	if (Type == "FC") {
		this->MeanShape[3] = Former.shape[3];
		this->VarianceShape[3] = Former.shape[3];
	}
	if (Type == "CONV") {
		this->MeanShape[1] = Former.shape[1];
		this->VarianceShape[1] = Former.shape[1];
	}
}

Activate::BatchNorm::~BatchNorm()
{
	if (InitParams) {
		cudaFree(MovingMean);
		cudaFree(MovingVariance);
		cudaFree(this->Scale);
		cudaFree(this->Offset);
	}
}
