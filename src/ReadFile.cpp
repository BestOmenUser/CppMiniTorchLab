#include"ReadFile.h"

ReadFile::ReadFile(const std::string _path, const std::string _type):path(_path),type(_type)
{
}

size_t ReadFile::ReserveBinary(int num) {
	int num1, num2, num3, num4;
	num1 = num& 255;
	num2 = (num >> 8) & 255;
	num3 = (num >> 16) & 255;
	num4 = (num >> 24) & 255;
	return ((num1 << 24) + (num2 << 16) + (num3 << 8) + num4);
}

ReadFile::~ReadFile() {
	if (infile.is_open()) {
		infile.close();
	}
	if (!block) {
		cudaFree(this->block);
	}
	if (!Data) {
		cudaFree(this->Data);
	}
}

int ReadFile::ReadImage()
{

	int magicNum;
	int imageNums;
	int Width;
	int Height;
	int Encode;
	size_t Chanels=1;
	infile.read((char*)&magicNum, sizeof(magicNum)); //00000803被取为03080000
	magicNum = ReserveBinary(magicNum);
	Encode = magicNum >> 8;
	infile.read((char*)&imageNums, sizeof(imageNums));
	imageNums = ReserveBinary(imageNums);
	infile.read((char*)&Width, sizeof(Width));
	Width = ReserveBinary(Width);
	infile.read((char*)&Height, sizeof(Height));
	Height = ReserveBinary(Height);
	this->shape = {(size_t) imageNums,Chanels,(size_t)Width,(size_t)Height };
	cudaMallocManaged(&this->Data, (size_t)(imageNums * Chanels * Width * Height +1)* sizeof(double));
	unsigned char* pixel;
	cudaMallocManaged(&pixel,sizeof(unsigned char));
	size_t index = -1;
	Length = imageNums * Chanels * Width * Height;
	while (!infile.eof()) {
		infile.read((char*)pixel, 1);
		Data[(++index)] = (double)(*pixel);
	}
	if (index < Length) {
		std::cout <<type<< " Read Data Fail" << std::endl;
		cudaFree(pixel);
		return 0;
	}
	cudaFree(pixel);
	return 1;
}


int ReadFile::ReadLabel()
{
	int magicNum;
	int LabelNums;
	int Encode;

	infile.read((char*)&magicNum, sizeof(magicNum)); //00000803被取为03080000
	magicNum = ReserveBinary(magicNum);
	Encode = magicNum >> 8;
	infile.read((char*)&LabelNums, sizeof(LabelNums));
	LabelNums = ReserveBinary(LabelNums);
	this->shape = {(size_t)LabelNums,1,1,1 };
	cudaMallocManaged(&this->Data,(size_t)(LabelNums+1)* sizeof(double));
	unsigned char* Label;
	cudaMallocManaged(&Label, sizeof(unsigned char));
	int index = -1;
	Length =LabelNums;

	while (!infile.eof()) {
		infile.read((char*)Label, 1);
		Data[++index] = (double)(*Label);
	}
	if (index < Length) {
		std::cout << type << " Read Data Fail" << std::endl;
		cudaFree(Label);
		return 0;
	}
	cudaFree(Label);
	return 1;
}


void ReadFile::LoadData()
{
	infile.open(this->path, std::ios::binary);
	if (!infile.is_open()) {
		std::cout << "open file fail" << std::endl;
	}
	if (type == "TrainImage" ||type== "TestImage") {
		if (!ReadImage()) {
			exit(-1);
		}
	}
	if (type == "TrainLabel" || type == "TestLabel") {
		if (!ReadLabel()) {
			exit(-1);
		}
	}
	cudaMallocManaged(&this->block, sizeof(DataBlock));
	(*block).Init(this->Data, this->shape);
	if (infile.is_open()) {
		infile.close();
	}
}

DataBlock& ReadFile::GetData()
{
	return *block;
}



void DataBlock::Init(double* _Data, std::vector<size_t> _shape)
{
	this->Data = _Data;
	this->shape=_shape;
}



double* DataBlock::GetData()
{
	return this->Data;
}

std::vector<size_t>* DataBlock::GetShape()
{
	return &(this->shape);
}
