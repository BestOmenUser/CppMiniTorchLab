#pragma once
#include"Activate.h"
#include"math.h"
#include"time.h"
#include"omp.h"
#include"chrono"


typedef void(*SetWeightPointer)(double*  Weight,shape Shape);
typedef void(*SetOptimizePointer)(double*& Data, shape DataShape, double*& Label, shape LabelShape,
	double*& SelectedData, shape &SelectedDataShape, double*& SelectedLabel, shape &SelectedLabelShape
	, size_t batch,size_t& Offset);


class Model {
public:
	Model() = default;
	template<class T,typename... Args>
	Model(T t,Args&&...args);
	DataBlock Forward(DataBlock data, DataBlock Label, bool Train = false);
	void BackPropagation();
	void Update(double learningRate);
	void Train(DataBlock& Data,DataBlock& Label,double learningRate=0.01, unsigned int epochs = 10, size_t batch = 0);
	void Evaluate(DataBlock& Data, DataBlock& Label);
	void ComputeGraph(shape Shape);
	template<class T>
	void LossFunction(T t);
	void SetWeight();
	void SetWeightFunc(SetWeightPointer WeightInitWay);
	void SetOptimizeFunc(SetOptimizePointer OptimizeInitWay);
	void SetNormal(Norm* NormFunc);
	void SetWeightDecay(NormNumberBase* Norm);
	~Model();
private:
	void DestroyInit();
	void SetNormalNumber();
	ActivateLink Layers;
	SetWeightPointer WeightFunc=NULL;
	SetOptimizePointer OptimizeFunc = NULL;
	Func* loss=NULL;
	Norm* norm=NULL;
	NormNumberBase* ModelNormNumber = NULL;
};


template<class T,typename... Args>
Model::Model(T t,Args&&...args)
{
	Layers.Push(t, args...);
}

template<class T>
void Model:: LossFunction(T t) {
	loss = t;
	Layers.Push(t);
}