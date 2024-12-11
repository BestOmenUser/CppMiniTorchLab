#pragma once
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
typedef std::vector<size_t> shape;
extern "C" size_t GetLength(shape Shape);

class DataBlock;

class ReadFile {
public:
	ReadFile() = default;
	ReadFile(const std::string _path,const std::string _type);
	void LoadData();
	DataBlock&  GetData ();
	~ReadFile();
private:
	size_t  ReserveBinary(int num);
	int ReadImage();
	int ReadLabel();
	std::string path;
	std::ifstream infile;
	std::vector<size_t> shape;
	std::string type;
	size_t Length;
	double* Data=nullptr;
	DataBlock* block=nullptr;
};

class DataBlock {
public:
	DataBlock() = default;
	void Init(double* _Data,std::vector<size_t> _shape);
	double* GetData();
	std::vector<size_t>* GetShape();
	friend class Model;
	friend class ReLU;
	double* Data;
	std::vector<size_t> shape;
private:
};

