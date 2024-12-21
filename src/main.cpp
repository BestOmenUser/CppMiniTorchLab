#include<iostream>
#include <crtdbg.h>
#include"NeuralNetwork.h"
#define _CRTDBG_MAP_ALLOC
const std::string TrainImagePath = "../FashionMNIST/raw/train-images.idx3-ubyte";
const std::string TestImagePath = "../FashionMNIST/raw/t10k-images.idx3-ubyte";
const std::string TestLabelPath = "../FashionMNIST/raw/t10k-labels.idx1-ubyte";
const std::string TrainLabelPath = "../FashionMNIST/raw/train-labels.idx1-ubyte";


int main()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);  // 获取设备属性
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
	ReadFile TrainLabel(TrainLabelPath, "TrainLabel");
	ReadFile TestLabel(TestLabelPath, "TestLabel");
	ReadFile TestImage(TestImagePath, "TestImage");
	ReadFile TrainImage(TrainImagePath, "TrainImage");
	TrainImage.LoadData();
	TrainLabel.LoadData();
	TestLabel.LoadData();
	TestImage.LoadData();
	Model model(new Layer::Conv2d(1, 3, 9, 9, 1, 3, 3),
		new Activate::Sigmoid,
		new Layer::MaxPooling(2, 2, 2),
		new Activate::ReLU,
		new Layer::Conv2d(3, 16, 4, 4),
		new Layer::Flatten,
		//new Activate::BatchNorm(1, 0, 0.9),
		new Layer::Linear(10 * 10 * 16, 512),
		//new Layer::DropOut(0.5),
		new Layer::Linear(512, 64),
		new Activate::ReLU,
		new Layer::Linear(64, 10),
		new Activate::SoftMax);
	model.LossFunction(new Loss::CrossEntropyLoss);
	model.SetNormal(new Normal::MinMax);
	model.SetWeightFunc(WeightInit::Xavier);
	model.SetWeightDecay(new NormNumber::L2NormNumber(0.01));
	model.SetOptimizeFunc(OpitmizeInit::MBgd);
	model.SetWeight();
	model.ComputeGraph(TestImage.GetData().shape);
	TrainImage.GetData().shape[0] = 1000;
	TrainLabel.GetData().shape[0] = 1000;
	model.Train(TrainImage.GetData(), TrainLabel.GetData(), 0.0000001, 10,256);
	model.Evaluate(TestImage.GetData(), TestLabel.GetData());

	shape v1 = { 256,16,25,26 };
	shape v2 = { 32,16,6,7 };
	shape v3 = { 256,26,100,42 };
	shape v4 = { 256,16,10,10 };
	double* a, * b;
	cudaMallocManaged(&a, GetLength(v1)*sizeof(double));
	cudaMallocManaged(&b, GetLength(v3) * sizeof(double));
	MatrixOperator::SetValue(a, v1, 0);
	MatrixOperator::MakeConvMatrix(a, v1, b, v3, v2, v4, 2);

	return 0;
}
