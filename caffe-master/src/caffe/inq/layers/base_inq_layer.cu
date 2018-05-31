#include "inq/base_inq_layer.hpp"
#include "inq/base_inq_layer.cuh"

namespace caffe {

template <typename Dtype>
void BaseInqLayer<Dtype>::QuantizeLayerInputs_gpu(
    Dtype* data, const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      Trim2FixedPoint_gpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, bw_layer_in_, rounding_, fl_layer_in_);
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_gpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
void BaseInqLayer<Dtype>::QuantizeLayerOutputs_gpu(Dtype* data,
      const int count) {
  switch (precision_) {
    case QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS:
      Trim2FixedPoint_gpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
    case QuantizationParameter_Precision_DYNAMIC_FIXED_POINT:
      Trim2FixedPoint_gpu(data, count, bw_layer_out_, rounding_, fl_layer_out_);
      break;
    case QuantizationParameter_Precision_MINIFLOAT:
      Trim2MiniFloat_gpu(data, count, fp_mant_, fp_exp_, rounding_);
      break;
    default:
      LOG(FATAL) << "Unknown trimming mode: " << precision_;
      break;
  }
}

template <typename Dtype>
__global__ void Trim2FixedPoint_kernel(Dtype* data, const int cnt,
      const int bit_width, const int rounding, const int fl) {
	CUDA_KERNEL_LOOP(index, cnt) {
    // Saturate data
    Dtype max_data = (powf(2, bit_width - 1) - 1) * powf(2, -fl);
    Dtype min_data = -powf(2, bit_width - 1) * powf(2, -fl);
    data[index] = fmax(fmin(data[index], max_data), min_data);
    // Round data
    data[index] /= powf(2, -fl);
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      data[index] = rint(data[index]);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      data[index] = __float2int_rd(data[index] + RandUniform_device(index));
      break;
    default:
      break;
    }
    data[index] *= powf(2, -fl);
	}
}

template <typename Dtype>
void BaseInqLayer<Dtype>::Trim2FixedPoint_gpu(Dtype* data, const int cnt,
      const int bit_width, const int rounding, int fl) {
  Trim2FixedPoint_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bit_width, rounding, fl);
}

template <typename Dtype>
__global__ void Trim2MiniFloat_kernel(Dtype* data, const int cnt,
      const int bw_mant, const int bw_exp, const int rounding){
	CUDA_KERNEL_LOOP(index, cnt) {
    Trim2MiniFloat_device(&data[index], bw_mant, bw_exp, rounding, index);
	}
}

template <typename Dtype>
void BaseInqLayer<Dtype>::Trim2MiniFloat_gpu(Dtype* data,
      const int cnt, const int bw_mant, const int bw_exp, const int rounding) {
  Trim2MiniFloat_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, bw_mant, bw_exp, rounding);
}

template <typename Dtype>
__global__ void Trim2IntegerPowerOf2_kernel(Dtype* data, const int cnt,
      const int min_exp, const int max_exp, const int rounding) {
	CUDA_KERNEL_LOOP(index, cnt) {
    float exponent = log2f(fabs((float)data[index]));
    int sign = data[index] >= 0 ? 1 : -1;
    switch (rounding) {
    case QuantizationParameter_Rounding_NEAREST:
      exponent = rint(exponent);
      break;
    case QuantizationParameter_Rounding_STOCHASTIC:
      exponent = __float2int_rd(exponent + RandUniform_device(index));
      break;
    default:
      break;
    }
    exponent = fmaxf(fminf(exponent, max_exp), min_exp);
    data[index] = sign * powf(2, exponent);
	}
}

template <typename Dtype>
void BaseInqLayer<Dtype>::Trim2IntegerPowerOf2_gpu(Dtype* data,
      const int cnt, const int min_exp, const int max_exp, const int rounding) {
  Trim2IntegerPowerOf2_kernel<<<CAFFE_GET_BLOCKS(cnt), CAFFE_CUDA_NUM_THREADS>>>(
      data, cnt, min_exp, max_exp, rounding);
}

// Explicit instantiations
template void BaseInqLayer<double>::QuantizeLayerInputs_gpu(double* data,
    const int count);
template void BaseInqLayer<float>::QuantizeLayerInputs_gpu(float* data,
    const int count);
template void BaseInqLayer<double>::QuantizeLayerOutputs_gpu(
    double* top_data, const int top_count);
template void BaseInqLayer<float>::QuantizeLayerOutputs_gpu(
    float* top_data, const int top_count);
template void BaseInqLayer<double>::Trim2FixedPoint_gpu(double* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseInqLayer<float>::Trim2FixedPoint_gpu(float* data,
    const int cnt, const int bit_width, const int rounding, int fl);
template void BaseInqLayer<double>::Trim2MiniFloat_gpu(double* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseInqLayer<float>::Trim2MiniFloat_gpu(float* data,
    const int cnt, const int bw_mant, const int bw_exp, const int rounding);
template void BaseInqLayer<double>::Trim2IntegerPowerOf2_gpu(double* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);
template void BaseInqLayer<float>::Trim2IntegerPowerOf2_gpu(float* data,
    const int cnt, const int min_exp, const int max_exp, const int rounding);

}  // namespace caffe


