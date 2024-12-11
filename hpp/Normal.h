#pragma once
#include"ReadFile.h"
#include"Tool.h"
typedef std::vector<size_t> shape;
class Norm {
public:
	Norm() = default;
	virtual void Produce(DataBlock& Data)=0;
};

namespace Normal {
	class MinMax;
}

class Normal::MinMax :public Norm {
public:
	MinMax() = default;
	virtual void Produce(DataBlock& Data);
};


class NormNumberBase {
public:
	NormNumberBase() = default;
	NormNumberBase(double Lamb) { this->Lambda = Lamb; }
	virtual void NormLoss(double* Weight,shape Shape,double& Loss)=0;
	virtual void NormBackPropagation(double* Weight,shape Shape,double* NormBP)=0;
	virtual ~NormNumberBase() =0 {};
private:
	double Lambda;
};

namespace NormNumber {
	class L1NormNumber;
	class L2NormNumber;
}

class NormNumber::L1NormNumber :public NormNumberBase {
	public:
		L1NormNumber() = default;
		L1NormNumber(double Lamb) { this->Lambda = Lamb; }
		virtual void NormLoss(double* Weight, shape Shape, double& Loss);
		virtual void NormBackPropagation(double* Weight, shape Shape, double* NormBP);
		virtual ~L1NormNumber() {};
	private:
		double Lambda;
};

class NormNumber::L2NormNumber :public NormNumberBase {
public:
	L2NormNumber() = default;
	L2NormNumber(double Lamb) { this->Lambda = Lamb; }
	virtual void NormLoss(double* Weight, shape Shape, double& Loss) ;
	virtual void NormBackPropagation(double* Weight, shape Shape, double* NormBP) ;
	virtual ~L2NormNumber() {};
private:
	double Lambda;
};

namespace OpitmizeInit {
	void Sgd(double*& Data, shape DataShape, double*& Label, shape LabelShape,
		double*& SelectedData, shape& SelectedDataShape, double*& SelectedLabel, shape& SelectedLabelShape,
		size_t batch, size_t& Offset);
	void Bgd(double*& Data, shape DataShape, double*& Label, shape LabelShape,
		double*& SelectedData, shape& SelectedDataShape, double*& SelectedLabel, shape& SelectedLabelShape,
		size_t batch, size_t& Offset);
	void MBgd(double*& Data, shape DataShape, double*& Label, shape LabelShape,
		double*& SelectedData, shape& SelectedDataShape, double*& SelectedLabel, shape& SelectedLabelShape,
		size_t batch, size_t& Offset);
};