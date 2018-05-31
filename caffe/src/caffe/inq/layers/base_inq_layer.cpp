#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include <time.h>

#include "inq/base_inq_layer.hpp"

namespace caffe {

template <typename Dtype>
BaseInqLayer<Dtype>::BaseInqLayer() {
  // Initialize random number generator
  srand(time(NULL));
}

template <typename Dtype>
void BaseInqLayer<Dtype>::QuantizeLayerInputs_cpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      Trim2FixedPoint_cpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseInqLayer<Dtype>::QuantizeLayerOutputs_cpu(
      Dtype* data, const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      Trim2FixedPoint_cpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_cpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_cpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseInqLayer<Dtype>::Trim2FixedPoint_cpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  for (int index = 0; index < cnt; ++index) {
    // Saturate data
    Dtype max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl);
    Dtype min_data = -pow(2, bit_width - 1) * pow(2, -fl);
    data[index] = std::max(std::min(data[index], max_data), min_data);
    // Round data
    data[index] /= pow(2, -fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = round(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = floor(data[index] + RandUniform_cpu());
      break;
    default:
      break;
    }
    data[index] *= pow(2, -fl);
	}
}

typedef union {
  float d;
  struct {
    unsigned int mantisa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

template <typename Dtype>
void BaseInqLayer<Dtype>::Trim2MiniFloat_cpu(Dtype* data, const int cnt,
      const int bw_mant, const int bw_exp, const int rounding) {
  for (int index = 0; index < cnt; ++index) {
    int bias_out = pow(2, bw_exp - 1) - 1;
    float_cast d2;
    // This casts the input to single precision
    d2.d = (float)data[index];
    int exponent=d2.parts.exponent - 127 + bias_out;
    double mantisa = d2.parts.mantisa;
    // Special case: input is zero or denormalized number
    if (d2.parts.exponent == 0) {
      data[index] = 0;
      return;
    }
    // Special case: denormalized number as output
    if (exponent < 0) {
      data[index] = 0;
      return;
    }
    // Saturation: input float is larger than maximum output float
    int max_exp = pow(2, bw_exp) - 1;
    int max_mant = pow(2, bw_mant) - 1;
    if (exponent > max_exp) {
      exponent = max_exp;
      mantisa = max_mant;
    } else {
      // Convert mantissa from long format to short one. Cut off LSBs.
      double tmp = mantisa / pow(2, 23 - bw_mant);
      switch (rounding) {
      case QuantizationParameter_Rounding_NEAREST:
        mantisa = round(tmp);
        break;
      case QuantizationParameter_Rounding_STOCHASTIC:
        mantisa = floor(tmp + RandUniform_cpu());
        break;
      default:
        break;
      }
    }
    // Assemble result
    data[index] = pow(-1, d2.parts.sign) * ((mantisa + pow(2, bw_mant)) /
        pow(2, bw_mant)) * pow(2, exponent - bias_out);
	}
}

template <typename Dtype>
double BaseInqLayer<Dtype>::RandUniform_cpu(){
  return rand() / (RAND_MAX+1.0);
}

template BaseInqLayer<double>::BaseInqLayer();
template BaseInqLayer<float>::BaseInqLayer();
template void BaseInqLayer<double>::QuantizeLayerInputs_cpu(double* data,
    const int count);
template void BaseInqLayer<float>::QuantizeLayerInputs_cpu(float* data,
    const int count);
template void BaseInqLayer<double>::QuantizeLayerOutputs_cpu(double* data,
    const int count);
template void BaseInqLayer<float>::QuantizeLayerOutputs_cpu(float* data,
    const int count);
template void BaseInqLayer<double>::Trim2FixedPoint_cpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseInqLayer<float>::Trim2FixedPoint_cpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseInqLayer<double>::Trim2MiniFloat_cpu(double* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseInqLayer<float>::Trim2MiniFloat_cpu(float* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template double BaseInqLayer<double>::RandUniform_cpu();
template double BaseInqLayer<float>::RandUniform_cpu();

}  // namespace caffe
