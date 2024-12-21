#include"NeuralNetwork.h"


DataBlock Model::Forward(DataBlock data,DataBlock Label,bool Train)
{
	Node* temp = this->Layers.head;
	DataBlock trained;
	trained.Data = data.Data;
	trained.shape = data.shape;

	while (temp!=this->Layers.Tail) {
			trained = (*(*temp).func).Forward(trained,true);
			temp = (*temp).next;
	}
	if (temp == this->Layers.Tail) {
			(*(*temp).func).SetLabel(Label);
			trained = (*(*temp).func).Forward(trained, true);
			temp = (*temp).next;
	}
	return trained;
}

void Model::BackPropagation()
{
	Node* temp = this->Layers.Tail;
	DataBlock Dx;
	while (temp) {
		Dx = (*(*temp).func).BackPropagation(Dx);
		temp = (*temp).front;
	}
}

void Model::Update(double learningRate)
{
	Node* temp = this->Layers.head;
	while (temp) {
		(*(*temp).func).Update(learningRate);
		temp = (*temp).next;
	}
}



Model::~Model()
{
}

void Model::DestroyInit()
{
	Node* temp = this->Layers.head;
	while (temp) {
		(*(*temp).func).DestroyInit();
		temp = (*temp).next;
	}
}

void Model::SetNormalNumber()
{
	Node* temp = this->Layers.head;
	while (temp) {
		(*(*temp).func).SetNormNumber(this->ModelNormNumber);
		temp = (*temp).next;
	}
}

void Model::Train(DataBlock& Data, DataBlock& Label, double learningRate, unsigned int epochs, size_t batch_size)
{
	srand(time(nullptr));
	if (norm) {
		(*this->norm).Produce(Data);
	}
	this->SetNormalNumber();
	this->SetWeight();
	auto before=std::chrono::system_clock::now();
	DataBlock data, label;
	size_t offset = 0;
	DataBlock output;
	for (unsigned int i = 0; i < epochs; i++) {
		std::cout << "epoch " << i + 1 << ":" << std::endl;
		offset = 0;
		while (offset != Label.shape[0]) {
			(*this->OptimizeFunc)(Data.Data, Data.shape, Label.Data, Label.shape,
				data.Data, data.shape, label.Data, label.shape, batch_size, offset);
			output=this->Forward(data, label, true);
			this->BackPropagation();
			this->Update(learningRate);
		}
		ShowEvaluate();
	}
	auto end= std::chrono::system_clock::now();
	auto diff = std::chrono::duration_cast<std::chrono::seconds>(end - before);
	std::cout <<"train spend "<< diff.count() <<"s"<< std::endl;
	DestroyInit();
}

void Model::Evaluate(DataBlock& Data, DataBlock& Label)
{
	if (norm) {
		(*this->norm).Produce(Data);
	}
	DataBlock data, label;
	size_t offset = 0;
	DataBlock output;
	offset = 0;
	std::cout << "Predict Evaluate:" << std::endl;
	while (offset != Label.shape[0]) {
		OpitmizeInit::MBgd(Data.Data, Data.shape, Label.Data, Label.shape,
				data.Data, data.shape, label.Data, label.shape, 64, offset);
			output = this->Forward(data, label, false);
		}
	ShowEvaluate();
	DestroyInit();
}

void Model::ComputeGraph(shape Shape)
{
	shape LocalShape=Shape;
	LocalShape[0] = 1;
	Node* temp = this->Layers.head;
	std::cout << "Model{" << std::endl;
	while (temp) {
		LocalShape=(*(*temp).func).ComputeGraph(LocalShape);
		temp = (*temp).next;
	}
	std::cout << "}" << std::endl;
}


void Model::SetWeight()
{
	Node* temp = this->Layers.head;
	while (temp) {
		(*(*temp).func).SetWeight(this->WeightFunc);
		temp = (*temp).next;
	}
}

void Model::SetWeightFunc(SetWeightPointer WeightInitWay)
{
	this->WeightFunc = WeightInitWay;
}

void Model::SetOptimizeFunc(SetOptimizePointer OptimizeInitWay)
{
	this->OptimizeFunc = OptimizeInitWay;
}

void Model::SetNormal(Norm* NormFunc)
{
	this->norm = NormFunc;
}

void Model::SetWeightDecay(NormNumberBase* Norm)
{
	this->ModelNormNumber = Norm;
}



