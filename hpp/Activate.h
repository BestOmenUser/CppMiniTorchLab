#pragma once
#include"Normal.h"
#include"Tool.h"
typedef void(*SetWeightPointer)(double* Weight, shape Shape);
void ShowEvaluate();



class Func {
public:
	Func() = default;
	virtual DataBlock Forward(DataBlock& Data,bool Train=false)=0;
	virtual DataBlock BackPropagation(DataBlock& Gradient) = 0;
	virtual void Update(double learningRate) = 0;
	virtual void SetWeight(SetWeightPointer WeightInitWay) = 0;
	virtual void SetLabel(DataBlock& Label)=0;
	virtual void Init() = 0;
	virtual shape ComputeGraph(shape Shape) = 0;
	virtual void DestroyInit() = 0;
	virtual void SetNormNumber(NormNumberBase* Norm) = 0;
	virtual void SuitData(DataBlock& Former)=0;
	virtual void SetZero() = 0;
	virtual ~Func() = 0 {};
protected:
	std::string Type;
	NormNumberBase* LayerNormNumber = NULL;
	bool Status_Init = false;
	DataBlock Input;
	double* Output;
	std::vector<size_t> OutShape;
	double* Do_Di;
	std::vector<size_t> Do_Di_Shape;
};


namespace Activate {
	class ReLU;
	class SoftMax;
	class Sigmoid;
	class BatchNorm;
}

namespace Layer {
	class Linear;
	class Flatten;
	class Conv2d;
	class MaxPooling;
	class AvgPooling;
	class DropOut;
}

namespace Loss {
	class MSE;
	class CrossEntropyLoss;
}


struct Node{
	Func* func;
	Node* next;
	Node* front;
};

class ActivateLink
{
public:
	ActivateLink();
	~ActivateLink();
	template<class T, class...Args>
	void Push(T layer, Args&&...args);
	template<class T>
	void Push(T layer);
	
	friend class Model;
private:
	Node* head = NULL;
	Node* Tail=NULL;
};

template<class T,class...Args>
void ActivateLink::Push(T layer, Args&&...args)
{
	if (layer) {
		if (!head) {
			head = new Node;
			Tail = new Node;
			(*head).func = layer;
			(*head).front = NULL;
			(*head).next = NULL;
			Tail = head;
		}
		else {
			Node* node = new Node;
			(*node).func = layer;
			(*node).next = NULL;
			(*node).front = Tail;
			(*Tail).next = node;
			Tail = node;
		}
		this->Push(args...);
	}
}

template<class T>
void ActivateLink::Push(T layer)
{
	if (layer) {
		if (!head) {
			head = new Node;
			Tail = new Node;
			(*head).func = layer;
			(*head).front = NULL;
			(*head).next = NULL;
			Tail = head;
		}
		else {
			Node* node = new Node;
			(*node).func = layer;
			(*node).next = NULL;
			(*node).front = Tail;
			(*Tail).next = node;
			Tail = node;
		}
	}
}


class Activate::ReLU:public Func{
public:
	ReLU() = default;
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void Update(double learningRate) {};
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void SetNormNumber(NormNumberBase* Norm) {};
	virtual void SetWeight(SetWeightPointer WeightInitWay) {};
	virtual ~ReLU() override;
private:
};


class Activate::BatchNorm :public Func {
public:
	BatchNorm() = delete;
	BatchNorm(double Scale,double Offset,double Momentum);
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void Update(double learningRate);
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void SetNormNumber(NormNumberBase* Norm) {};
	virtual void SetWeight(SetWeightPointer WeightInitWay) {};
	virtual ~BatchNorm() override;
private:
	double *Scale;
	double* ScaleGradient;
	double* OffsetGradient;
	double *Offset;
	double ScaleNumber;
	double OffsetNumber;
	double* Mean;
	//double* MeanGradient;
	shape MeanShape;
	double* Variance;
	//double* VarianceGradient;
	shape VarianceShape;
	double* InputHat;
	double* MovingMean;
	double* MovingVariance;
	double Momentum;
	bool InitParams=false;
	std::string Type;
};

class Layer::DropOut :public Func {
public:
	DropOut() = delete;
	DropOut(double Rate);
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void Update(double learningRate) {};
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void SetNormNumber(NormNumberBase* Norm) {};
	virtual void SetWeight(SetWeightPointer WeightInitWay) {};
	virtual ~DropOut() override;
private:
	double Rate;
};

class Activate::Sigmoid :public Func {
public:
	Sigmoid() = default;
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void Update(double learningRate) {};
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void SetNormNumber(NormNumberBase* Norm) {};
	virtual void SetWeight(SetWeightPointer WeightInitWay) {};
	virtual ~Sigmoid() override;
private:
};



class Layer::Linear :public Func {
public:
	bool Status_Init = false;
	Linear () = default;
	Linear(size_t inputSize, size_t outputSize);
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void SetWeight(SetWeightPointer WeightInitWay);
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void Update(double learningRate);
	virtual void SetNormNumber(NormNumberBase* Norm);
	virtual ~Linear();
private:
	double* Weight;
	std::vector<size_t> WeightShape;
	double* Bias;
	std::vector<size_t> BiasShape;
	double* WeightTemp;
	double* BiasTemp;
};


class Layer::Conv2d:public Func {
public:
	bool Status_Init = false;
	Conv2d() = default;
	Conv2d(size_t inputChanell, size_t outputChanel,size_t kernelSizeWidth, size_t kernelSizeHeight,
		size_t stride=1,size_t HorizontallyPadding =0, size_t VerticallyPadding=0);
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void SetWeight(SetWeightPointer WeightInitWay);
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void Update(double learningRate);
	virtual void SetNormNumber(NormNumberBase* Norm);
	virtual ~Conv2d();
private:
	void MakePadding();
	DataBlock ProcessedData;
	double* Kernel;
	std::vector<size_t>KernelShape;
	double* Bias;
	std::vector<size_t> BiasShape;
	double* KernelTemp;
	double* BiasTemp;
	size_t Stride;
	size_t HorizontallyPadding;
	size_t VerticallyPadding;
};


class Layer::MaxPooling :public Func {
public:
	bool Status_Init = false;
	MaxPooling() = default;
	MaxPooling(size_t kernelSizeWidth, size_t kernelSizeHeight,
		size_t stride = 1, size_t HorizontallyPadding = 0, size_t VerticallyPadding = 0);
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void SetWeight(SetWeightPointer WeightInitWay) {}
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void Update(double learningRate){}
	virtual void SetNormNumber(NormNumberBase* Norm){}
	virtual ~MaxPooling();
private:
	void MakePadding();
	DataBlock ProcessedData;
	double* Kernel;
	std::vector<size_t>KernelShape;
	size_t Stride;
	size_t HorizontallyPadding;
	size_t VerticallyPadding;
};


class Layer::AvgPooling :public Func {
public:
	bool Status_Init = false;
	AvgPooling() = default;
	AvgPooling(size_t kernelSizeWidth, size_t kernelSizeHeight,
		size_t stride = 1, size_t HorizontallyPadding = 0, size_t VerticallyPadding = 0);
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void SetZero();
	virtual void SetWeight(SetWeightPointer WeightInitWay) {}
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SuitData(DataBlock& Former);
	virtual void Update(double learningRate) {}
	virtual void SetNormNumber(NormNumberBase* Norm) {}
	virtual ~AvgPooling();
private:
	void MakePadding();
	DataBlock ProcessedData;
	double* Kernel;
	std::vector<size_t>KernelShape;
	size_t Stride;
	size_t HorizontallyPadding;
	size_t VerticallyPadding;
};



class Activate::SoftMax:public Func {
public:
	SoftMax() = default;
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock&  Gradient);
	virtual void Update(double learningRate) {}
	virtual void SetWeight(SetWeightPointer WeightInitWay) {}
	virtual void SetLabel(DataBlock& Label) {}
	virtual void Init();
	virtual shape ComputeGraph(shape Shape) ;
	virtual void DestroyInit();
	virtual void SetNormNumber(NormNumberBase* Norm) {}
	virtual void SuitData(DataBlock& Former);
	virtual void SetZero();
	virtual ~SoftMax(){};
};



class Loss::MSE:public Func {
public:
	MSE() = default;
	virtual DataBlock Forward(DataBlock& Data,bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit() ;
	virtual void SuitData(DataBlock& Former) ;
	virtual void SetWeight(SetWeightPointer WeightInitWay) {}
	virtual void SetZero();
	virtual void Update(double learningRate) {}
	virtual void SetLabel(DataBlock& Label);
	virtual void SetNormNumber(NormNumberBase* Norm);
	virtual ~MSE() override;
private:
	DataBlock Label;
};


class Loss::CrossEntropyLoss:public Func {
public:
	CrossEntropyLoss() = default;
	virtual DataBlock Forward(DataBlock& Data, bool Train = false);
	virtual DataBlock BackPropagation(DataBlock& Gradient);
	virtual void Update(double learningRate) {};
	virtual void SetWeight(SetWeightPointer WeightInitWay) {};
	virtual void SetLabel(DataBlock& Label) ;
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SetNormNumber(NormNumberBase* Norm) {};
	virtual void SuitData(DataBlock& Former) ;
	virtual void SetZero();
	virtual ~CrossEntropyLoss() override;
private:
	void MakeOneHot(size_t KindNums);
	void GetAccuracy(DataBlock Label,DataBlock Predicted,bool Trian);
	bool LabelInit = false;
	DataBlock OneHot;
	DataBlock Label;
};



class Layer::Flatten :public Func {
public:
	Flatten() = default;
	virtual DataBlock Forward(DataBlock& data, bool train = false);
	virtual DataBlock BackPropagation(DataBlock& data);
	virtual void SetLabel(DataBlock& Label) {};
	virtual void Init();
	virtual shape ComputeGraph(shape Shape);
	virtual void DestroyInit();
	virtual void SetZero();
	virtual void SuitData(DataBlock& Former);
	virtual void SetWeight(SetWeightPointer WeightInitWay) {}
	virtual void Update(double learningRate) {};
	virtual void SetNormNumber(NormNumberBase* Norm) {}
	virtual ~Flatten();
private:
};
