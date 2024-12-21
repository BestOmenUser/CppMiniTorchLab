#include "Normal.h"
void Normal::MinMax::Produce(DataBlock& Data)
{
    size_t size = 1;
    double average=0;
    double min=9999.9, max=-9999.9;
    for (size_t i : Data.shape) {
        size *= i;
    }
    for (size_t i = 0; i < size; i++) {
        if (min > Data.Data[i]) {
            min = Data.Data[i];
        }
        if (max < Data.Data[i]) {
            max = Data.Data[i];
        }
    }
    double diff = max - min;
    for (size_t i = 0; i < size; i++) {
        Data.Data[i] = (Data.Data[i]-min) / diff;
    }
}

void NormNumber::L1NormNumber::NormLoss(double* Weight, shape Shape, double& Loss)
{
}

void NormNumber::L1NormNumber::NormBackPropagation(double* Weight, shape Shape, double* NormBP)
{
    L1Operator::L1Backward(Weight, Shape, NormBP, this->Lambda);
}

void NormNumber::L2NormNumber::NormLoss(double* Weight, shape Shape, double& Loss)
{
}

void NormNumber::L2NormNumber::NormBackPropagation(double* Weight, shape Shape, double* NormBP)
{
    MatrixOperator::MultiplyNumber(Weight, Shape, NormBP,this->Lambda);
}

void OpitmizeInit::Sgd(double*& Data, shape DataShape, double*& Label, shape LabelShape,
    double*& SelectedData, shape &SelectedDataShape, double*& SelectedLabel, shape &SelectedLabelShape, size_t batch, size_t& Offset)
{
    size_t selected = rand() % LabelShape[0];
    SelectedDataShape = DataShape;
    SelectedDataShape[0] = 1;
    SelectedLabelShape = LabelShape;
    SelectedLabelShape[0] = 1;
    SelectedData = Data + selected* GetLength(DataShape) / DataShape[0];
    SelectedLabel = Label + selected*GetLength(LabelShape) / LabelShape[0];
    Offset = LabelShape[0];
}

void OpitmizeInit::Bgd(double*& Data, shape DataShape, double*& Label, shape LabelShape, double*& SelectedData,
    shape &SelectedDataShape, double*& SelectedLabel, shape &SelectedLabelShape, size_t batch, size_t& Offset)
{
    Offset = DataShape[0];
    SelectedDataShape = DataShape;
    SelectedLabelShape = LabelShape;
    SelectedData = Data;
    SelectedLabel = Label;
}

void OpitmizeInit::MBgd(double*& Data, shape DataShape, double*& Label, shape LabelShape, double*& SelectedData,
    shape &SelectedDataShape, double*& SelectedLabel, shape &SelectedLabelShape, size_t batch, size_t& Offset)
{
    if (Offset + batch >= DataShape[0]) {
        batch = DataShape[0] - Offset;
    }
    SelectedDataShape = DataShape;
    SelectedLabelShape = LabelShape;
    SelectedDataShape[0] = batch;
    SelectedLabelShape[0] = batch;
    SelectedData = Data + (Offset*GetLength(DataShape)/DataShape[0]);
    SelectedLabel = Label + (Offset*GetLength(LabelShape)/LabelShape[0]);
    Offset += batch;
}
