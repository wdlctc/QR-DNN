#include "boost/algorithm/string.hpp"

#include "caffe/caffe.hpp"
#include "inq/INQuantization.hpp"
#include "caffe/util/signal_handler.h"
#include <algorithm>

using caffe::Caffe;
using caffe::Net;
using caffe::string;
using caffe::vector;
using caffe::Blob;
using caffe::LayerParameter;
using caffe::NetParameter;

using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using std::ostringstream;

INQuantization::INQuantization(string model, string test_model, string weights, string model_quantized,
    int iterations, string trimming_mode, string gpus, string solver) {
  this->model_ = model;
  this->weights_ = weights;
  this->model_quantized_ = model_quantized;
  this->iterations_ = iterations;		
  this->trimming_mode_ = trimming_mode;
  this->gpus_ = gpus;
  this->solver_ = solver;
  if(test_model.size() == 0){
    this->test_model_ = model;
    LOG(INFO) << "Undefined Test_model, use model_ instead";
  }
  else{
     this->test_model_ = test_model;
  }
  
}

void INQuantization::INQuantizeNet() {
  CheckWritePermissions(model_quantized_);
  test_mode = false;
  caffe::SolverParameter solver_param;
  caffe::ReadSolverParamsFromTextFileOrDie(solver_, &solver_param);
  vector<int> gpus;
    // Parse GPU ids or use all available devices
  if (gpus_ == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus.push_back(i);
    }
  } else if (gpus_.size()) {
    vector<string> strings;
    boost::split(strings, gpus_, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus.push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus.size(), 0);
  }
  // Set device id and mode
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
	solver_param.set_device_id(gpus[0]);
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
	Caffe::set_solver_count(1);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  float accuracy;

  // Run the reference floating point network on validation set to find baseline
  // accuracy.
  //Net<float>* net_val = new Net<float>(test_model_, caffe::TEST);
  //net_val->CopyTrainedLayersFrom(weights_);
  //RunForwardBatches(this->iterations_, net_val, &accuracy, true);
  //test_score_baseline_ = accuracy;
  //delete net_val;

  // Run the reference floating point network on train data set to find maximum
  // values. Do statistic for 10 batches.
  Net<float>* net_test = new Net<float>(model_, caffe::TRAIN);
  net_test->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(100, net_test, &accuracy, true);
  delete net_test;
  // Do network quantization and scoring.

  if (trimming_mode_ == "dynamic_fixed_point") {
    Quantize2DynamicFixedPoint();
  } else if (trimming_mode_ == "minifloat") {
    Quantize2MiniFloat();
  } else if (trimming_mode_ == "integer_power_of_2_weights") {
    Quantize2IntegerPowerOf2Weights();
  } else {
	LOG(INFO) << "Undefine mode, use integer_power_of_2_weights instead ";
    Quantize2IntegerPowerOf2Weights();
  }


  caffe::SignalHandler signal_handler(
        caffe::SolverAction::STOP,
        caffe::SolverAction::SNAPSHOT);
  
  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

  solver->SetActionFunction(signal_handler.GetActionFunction());
  CopyLayers(solver.get(), weights_);


  LOG(INFO) << "LC TEST";
    solver->Snapshot(); 
  INQ(solver->net(), 0.25); 
  solver->Solve();
  solver->iter_ = 0;
  INQ(solver->net(), 0.5); 
  solver->Solve();
  solver->iter_ = 0;
  INQ(solver->net(), 0.75); 
  solver->Solve();
  solver->iter_ = 0;
  INQ(solver->net(), 0.875);  
  solver->Solve();
  solver->iter_ = 0;
  INQ(solver->net(), 0.9375);  
  solver->Solve();

  solver->iter_ = 9;
  INQ(solver->net(), 1); 
  solver->Snapshot(); 
  solver->TestAll();

  LOG(INFO) << "Optimization Done.";

}

void INQuantization::INQuantizeTest() {

  CheckWritePermissions(model_quantized_);
  test_mode = true;
  vector<int> gpus;
    // Parse GPU ids or use all available devices
  if (gpus_ == "all") {
    int count = 0;
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDeviceCount(&count));
#else
    NO_GPU;
#endif
    for (int i = 0; i < count; ++i) {
      gpus.push_back(i);
    }
  } else if (gpus_.size()) {
    vector<string> strings;
    boost::split(strings, gpus_, boost::is_any_of(","));
    for (int i = 0; i < strings.size(); ++i) {
      gpus.push_back(boost::lexical_cast<int>(strings[i]));
    }
  } else {
    CHECK_EQ(gpus.size(), 0);
  }
  // Set device id and mode
  if (gpus.size() != 0) {
    LOG(INFO) << "Use GPU with device ID " << gpus[0];
    Caffe::SetDevice(gpus[0]);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(INFO) << "Use CPU.";
    Caffe::set_mode(Caffe::CPU);
  }

  // Run the reference floating point network on validation set to find baseline
  // accuracy.
  Net<float>* net_val = new Net<float>(test_model_, caffe::TEST);
  net_val->CopyTrainedLayersFrom(weights_);
  float accuracy;
  RunForwardBatches(this->iterations_, net_val, &accuracy, true);
  test_score_baseline_ = accuracy;
  delete net_val;

  // Do network quantization and scoring.
  if (trimming_mode_ == "dynamic_fixed_point") {
    Quantize2DynamicFixedPoint();
  } else if (trimming_mode_ == "minifloat") {
    Quantize2MiniFloat();
  } else if (trimming_mode_ == "integer_power_of_2_weights") {
    Quantize2IntegerPowerOf2Weights();
  } else {
	LOG(INFO) << "Undefine mode, use Pruning_inspired_partition instead ";
    Quantize2IntegerPowerOf2Weights();
  }

}

void INQuantization::Trim2IntegerPowerOf2_cpu(float* data,
      const int cnt, const float thread, const int max_exp, const int min_exp,
      float* mask) {
  for (int index = 0; index < cnt; ++index) {
    if (fabs(data[index])  >= thread){
	mask[index] = 0;
        //data[index]=data[index]*0.9;
	float exponent = log2f((float)fabs(data[index]));
	exponent = floor(exponent);
	int sign = data[index] >= 0 ? 1 : -1;
	if (pow(2, exponent)*1.5 < sign * data[index] )
	exponent = exponent + 1;
	//exponent = std::max(std::min(exponent, (float)max_exp), (float)min_exp);
	exponent = std::min(exponent, (float)max_exp);
	data[index] = exponent < min_exp ? 0 :sign * pow(2, exponent);
    }
    else {
	mask[index] = 1;
	}
  }
}

void INQuantization::Trim2FixedPoint_cpu(float* data,
      const int cnt, const float thread, const int fl,
      float* mask) {
  for (int index = 0; index < cnt; ++index) {
    if (fabs(data[index])  >= thread){
	mask[index] = 0;
        //data[index]=data[index]*0.9;
        float max_data = (pow(2, 8 - 1) - 1) * pow(2, fl);
        float min_data = -pow(2, 8 - 1) * pow(2, fl);
        data[index] = std::max(std::min(data[index], max_data), min_data);
        // Round data
        data[index] /= pow(2, fl);
	data[index] = round(data[index]);
	data[index] *= pow(2, fl);
    }
    else {
	mask[index] = 1;
	}
  }
}


void INQuantization::INQ(caffe::shared_ptr<Net<float> > caffe_net, float per){
/*
  for (int param_id = 0; param_id <caffe_net->learnable_params().size(); ++param_id) {
    LOG(INFO) << "LC TEST" <<caffe_net->learnable_params()[param_id]->count();
  }
*/
  LOG(INFO) << "step:" << per;
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(model_, &param);
  for (int i = 0; i < param.layer_size(); ++i) {
    LOG(INFO) << "layer_names_[" << i <<"] = " << param.layer(i).name();
    //LOG(INFO) << "il_params[" << i << "] = " << GetIntegerLengthParams(param.layer(i).name());
    int il_param = GetIntegerLengthParams(param.layer(i).name());
    //int io_param = GetIntegerLengthOut(param.layer(i).name());

    if ( param.layer(i).type() == "Convolution" || param.layer(i).type() == "InnerProduct" ) {

        shared_ptr< Layer<float> > caffe_layer = 
	caffe_net->layer_by_name(param.layer(i).name());

        int count = caffe_layer->blobs()[0]->count();
        vector <float > blob_sorted;
        float * blob_pre = caffe_layer->blobs()[0]->mutable_cpu_data();
        for (int k = 0; k < count; ++k){
          blob_sorted.push_back( fabs(blob_pre [k]));
        }
        std::sort (blob_sorted.begin(), blob_sorted.end());
        float thread = blob_sorted[int(count * (1-per))];
        LOG(INFO) << "count = " << count;
        LOG(INFO) << "min param = " << blob_sorted[0];
        LOG(INFO) << "max param = " << blob_sorted[count-1];
        LOG(INFO) << "thread param = " << thread << " i = " << int(count * (1-per));
        int fl = il_param - 8;
        float max_data = (pow(2, 8 - 1) - 1) * pow(2, fl);
        float min_data = -pow(2, 8 - 1) * pow(2, fl);
        LOG(INFO) << "Fl = " << fl << " il = " << il_param;
        LOG(INFO) << "UP_LIMIT = " << max_data << " LOW_LIMIT = "<< min_data;
        //for (int k = 0; k < count; ++k) LOG(INFO) << "data = "<< blob_sorted[k];
        blob_sorted.clear();

        if (trimming_mode_ == "integer_power_of_2_weights") {
        
            Trim2IntegerPowerOf2_cpu(blob_pre , count , thread,
                                     il_param, il_param-7+1,
                                     caffe_layer->blobs()[0]->mutable_cpu_mask());
        }
        else if (trimming_mode_ == "dynamic_fixed_point") {
            
            Trim2FixedPoint_cpu(blob_pre , count , thread,
                                il_param-7,
                                caffe_layer->blobs()[0]->mutable_cpu_mask());
        } 
        
        
        //Trim2IntegerPowerOf2_cpu(blob_pre , count , thread,
        //il_param, il_param-7+1,
        //caffe_layer->blobs()[0]->mutable_cpu_mask());
/*
	if(caffe_layer->blobs().size()>1){
	        count = caffe_layer->blobs()[1]->count();
		blob_pre = caffe_layer->blobs()[1]->mutable_cpu_data();
		for (int k = 0; k < count; ++k){
		  blob_sorted.push_back( fabs(blob_pre [k]));
		}
		std::sort (blob_sorted.begin(), blob_sorted.end());
		float thread = blob_sorted[int(count * (1-per))];
		LOG(INFO) << "count = " << count;
		LOG(INFO) << "min param = " << blob_sorted[0];
		LOG(INFO) << "max param = " << blob_sorted[count-1];
		LOG(INFO) << "thread param = " << thread << " i = " << int(count * (1-per));
		//for (int k = 0; k < count; ++k) LOG(INFO) << "data = "<< blob_sorted[k];
		blob_sorted.clear();
		Trim2FixedPoint_cpu(blob_pre , count , thread, io_param,
		caffe_layer->blobs()[1]->mutable_cpu_mask());
		}
*/
    }
  }
}

void INQuantization::CheckWritePermissions(const string path) {
  std::ofstream probe_ofs(path.c_str());
  if (probe_ofs.good()) {
    probe_ofs.close();
    std::remove(path.c_str());
  } else {
    LOG(FATAL) << "Missing write permissions";
  }
}

// Load the weights from the specified caffemodel(s) into the train and
// test nets.
void INQuantization::CopyLayers(caffe::Solver<float>* solver, const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(",") );
  for (int i = 0; i < model_names.size(); ++i) {
    LOG(INFO) << "Finetuning from " << model_names[i];
    solver->net()->CopyTrainedLayersFrom(model_names[i]);
    for (int j = 0; j < solver->test_nets().size(); ++j) {
      solver->test_nets()[j]->CopyTrainedLayersFrom(model_names[i]);
    }
  }
}

void INQuantization::RunForwardBatches(const int iterations,
      Net<float>* caffe_net, float* accuracy, const bool do_stats,
      const int score_number) {
  LOG(INFO) << "Running for " << iterations << " iterations.";
  vector<Blob<float>* > bottom_vec;
  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < iterations; ++i) {
    float iter_loss;
    // Do forward propagation.
    const vector<Blob<float>*>& result =
        caffe_net->Forward(bottom_vec, &iter_loss);
    // Find maximal values in network.
    if(do_stats) {
      caffe_net->RangeInLayers(&layer_names_, &max_in_, &max_out_,
          &max_params_);
    }
    // Keep track of network score over multiple batches.
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = caffe_net->blob_names()[
            caffe_net->output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }
    }
  }
  loss /= iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net->blob_names()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net->blob_loss_weights()[
        caffe_net->output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }
  *accuracy = test_score[score_number] / iterations;
}


void INQuantization::Quantize2DynamicFixedPoint() {
  // Find the integer length for dynamic fixed point numbers.
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part.
  // For layer activations, we reduce the integer length by one bit.
  for (int i = 0; i < layer_names_.size(); ++i) {
    il_in_.push_back((int)ceil(log2(max_in_[i])));
    il_out_.push_back((int)ceil(log2(max_out_[i])));
    il_params_.push_back((int)ceil(log2(max_params_[i])+1));
  }
  // Debug
  for (int k = 0; k < layer_names_.size(); ++k) {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        "\ninteger length input=" << il_in_[k] <<
        "\ninteger length output=" << il_out_[k] <<
        "\ninteger length parameters=" << il_params_[k];
    LOG(INFO) << "Layer " << layer_names_[k] <<
        "\nfloat length input=" << max_in_[k] <<
        "\nfloat length output=" << max_out_[k] <<
        "\nfloat length parameters=" << max_params_[k];
  }

  // Score net with dynamic fixed point convolution parameters.
  // The rest of the net remains in high precision format.
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(test_model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  float accuracy;
  Net<float>* net_test;
  EditNetDescriptionDynamicFixedPoint(&param);
  // Bit-width of layer activations is hard-coded to 8-bit.
  net_test = new Net<float>(param);
  net_test->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(iterations_, net_test, &accuracy);
  delete net_test;

  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of dynamic fixed point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Dynamic fixed point net:";
  LOG(INFO) << "Accuracy: " << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

void INQuantization::Quantize2MiniFloat() {
  // Find the integer length for dynamic fixed point numbers.
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part.
  // For layer activations, we reduce the integer length by one bit.
  exp_bits_ = 4;
  for (int i = 0; i < layer_names_.size(); ++i) {
    int exp_in = ceil(log2(log2(max_in_[i]) - 1) + 1);
    int exp_out = ceil(log2(log2(max_out_[i]) - 1) + 1);
    exp_bits_ = std::max( std::max( exp_bits_, exp_in ), exp_out);
  }
  // Debug
  for (int k = 0; k < layer_names_.size(); ++k) {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", exp length input=" << ceil(log2(log2(max_in_[k]) - 1) + 1) <<
        ", exp length output=" << ceil(log2(log2(max_out_[k]) - 1) + 1) ;
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", float length input=" << max_in_[k] <<
        ", float length output=" << max_out_[k] <<
        ", float length parameters=" << max_params_[k];
  }    
    LOG(INFO) << "exp_bits_ = " << exp_bits_;

  // Score net with dynamic fixed point convolution parameters.
  // The rest of the net remains in high precision format.
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(test_model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  float accuracy;
  Net<float>* net_test;
  EditNetDescriptionMiniFloat(&param);
  // Bit-width of layer activations is hard-coded to 8-bit.
  net_test = new Net<float>(param);
  net_test->CopyTrainedLayersFrom(weights_);
  RunForwardBatches(iterations_, net_test, &accuracy);
  delete net_test;

  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of dynamic fixed point analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Convolutional (CONV) and fully";
  LOG(INFO) << "connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Minifloat net:";
  LOG(INFO) << "Accuracy: " << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

void INQuantization::Quantize2IntegerPowerOf2Weights() {
  // Find the integer length for dynamic fixed point numbers.
  // The integer length is chosen such that no saturation occurs.
  // This approximation assumes an infinitely long factional part.
  // For layer activations, we reduce the integer length by one bit.
  for (int i = 0; i < layer_names_.size(); ++i) {
    il_in_.push_back((int)floor(log2(max_in_[i]*4/3)));
    il_out_.push_back((int)floor(log2(max_out_[i]*4/3)));
    il_params_.push_back((int)floor(log2(max_params_[i]*4/3)));
  }
  // Debug
  for (int k = 0; k < layer_names_.size(); ++k) {
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", integer length input=" << il_in_[k] <<
        ", integer length output=" << il_out_[k] <<
        ", integer length parameters=" << il_params_[k];
    LOG(INFO) << "Layer " << layer_names_[k] <<
        ", float length input=" << max_in_[k] <<
        ", float length output=" << max_out_[k] <<
        ", float length parameters=" << max_params_[k];
  }
  // Score net with integer-power-of-two weights and dynamic fixed point
  // activations.
  NetParameter param;
  caffe::ReadNetParamsFromTextFileOrDie(test_model_, &param);
  param.mutable_state()->set_phase(caffe::TEST);
  float accuracy;
  Net<float>* net_test;
  EditNetDescriptionIntegerPowerOf2Weights(&param);

  if(test_mode){
    // Bit-width of layer activations is hard-coded to 8-bit.
    net_test = new Net<float>(param);
    net_test->CopyTrainedLayersFrom(weights_);
    RunForwardBatches(iterations_, net_test, &accuracy);
    delete net_test;
  }

  // Write prototxt file of quantized net
  param.release_state();
  WriteProtoToTextFile(param, model_quantized_);

  // Write summary of integer-power-of-2-weights analysis to log
  LOG(INFO) << "------------------------------";
  LOG(INFO) << "Network accuracy analysis for";
  LOG(INFO) << "Integer-power-of-two weights";
  LOG(INFO) << "in Convolutional (CONV) and";
  LOG(INFO) << "fully connected (FC) layers.";
  LOG(INFO) << "Baseline 32bit float: " << test_score_baseline_;
  LOG(INFO) << "Power-of-2 net:";
  LOG(INFO) << "4bit: \t" << accuracy;
  LOG(INFO) << "Please fine-tune.";
}

void INQuantization::EditNetDescriptionDynamicFixedPoint(
      NetParameter* param) {
  caffe::QuantizationParameter_Precision precision =
      caffe::QuantizationParameter_Precision_DYNAMIC_FIXED_POINT;
  caffe::QuantizationParameter_Rounding rounding =
      caffe::QuantizationParameter_Rounding_NEAREST;
  const int bw_in = 8;
  const int bw_out = 8;
  const int bw_conv = 8;
  for (int i = 0; i < param->layer_size(); ++i) {
    if ( param->layer(i).type() == "Convolution" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      if(test_mode) param_layer->set_type("ConvolutionInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_rounding_scheme(rounding);
      param_layer->mutable_quantization_param()->set_fl_layer_in(bw_in -
          GetIntegerLengthIn(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
      param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
           GetIntegerLengthOut(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
      param_layer->mutable_quantization_param()->set_fl_params(bw_conv - GetIntegerLengthParams(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_params(bw_conv);
    } 
    else if ( param->layer(i).type() == "InnerProduct" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      if(test_mode) param_layer->set_type("FcInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_rounding_scheme(rounding);
      param_layer->mutable_quantization_param()->set_fl_layer_in(bw_in -
          GetIntegerLengthIn(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
      param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
           GetIntegerLengthOut(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
      param_layer->mutable_quantization_param()->set_fl_params(bw_conv - GetIntegerLengthParams(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_params(bw_conv);
    } /*
      else if ( param->layer(i).type() == "Eltwise" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      if(test_mode) param_layer->set_type("EltwiseInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_rounding_scheme(rounding);
      param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
           GetIntegerLengthOut(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
    }*/

  }
}

void INQuantization::EditNetDescriptionMiniFloat(
      NetParameter* param) {
  caffe::QuantizationParameter_Precision precision =
      caffe::QuantizationParameter_Precision_MINIFLOAT;
  caffe::QuantizationParameter_Rounding rounding =
      caffe::QuantizationParameter_Rounding_NEAREST;

  const int bitwidth = 8;
  for (int i = 0; i < param->layer_size(); ++i) {
  
    if ( param->layer(i).type() == "Convolution" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      if(test_mode) param_layer->set_type("ConvolutionInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_rounding_scheme(rounding);
      param_layer->mutable_quantization_param()->set_mant_bits(bitwidth
          - exp_bits_ - 1);
      param_layer->mutable_quantization_param()->set_exp_bits(exp_bits_);

    } else if ( param->layer(i).type() == "InnerProduct" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      if(test_mode) param_layer->set_type("FcInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      param_layer->mutable_quantization_param()->set_rounding_scheme(rounding);
      param_layer->mutable_quantization_param()->set_mant_bits(bitwidth
          - exp_bits_ - 1);
      param_layer->mutable_quantization_param()->set_exp_bits(exp_bits_);
    }
  }
}

void INQuantization::EditNetDescriptionIntegerPowerOf2Weights(
      NetParameter* param) {
  caffe::QuantizationParameter_Precision precision =
      caffe::QuantizationParameter_Precision_INTEGER_POWER_OF_2_WEIGHTS;
  const int bw_in = 8;
  const int bw_out = 8;
  for (int i = 0; i < param->layer_size(); ++i) {
    if ( param->layer(i).type() == "Convolution" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("ConvolutionInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      // Weights are represented as 2^e where e in [-8,...,-1].
      // This choice of exponents works well for AlexNet.
      param_layer->mutable_quantization_param()->set_exp_min(GetIntegerLengthParams(param->layer(i).name())-7);
      param_layer->mutable_quantization_param()->set_exp_max(GetIntegerLengthParams(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_fl_layer_in(bw_in -
          GetIntegerLengthIn(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
      param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
           GetIntegerLengthOut(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
    } 
    else if ( param->layer(i).type() == "InnerProduct" ) {
      LayerParameter* param_layer = param->mutable_layer(i);
      param_layer->set_type("FcInq");
      param_layer->mutable_quantization_param()->set_precision(precision);
      // Weights are represented as 2^e where e in [-8,...,-1].
      // This choice of exponents works well for AlexNet.
      param_layer->mutable_quantization_param()->set_exp_min(GetIntegerLengthParams(param->layer(i).name())-7);
      param_layer->mutable_quantization_param()->set_exp_max(GetIntegerLengthParams(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_fl_layer_in(bw_in -
          GetIntegerLengthIn(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_in(bw_in);
      param_layer->mutable_quantization_param()->set_fl_layer_out(bw_out -
           GetIntegerLengthOut(param->layer(i).name()));
      param_layer->mutable_quantization_param()->set_bw_layer_out(bw_out);
    }
  }
}

int INQuantization::GetIntegerLengthParams(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_params_[pos];
}

int INQuantization::GetIntegerLengthIn(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_in_[pos];
}

int INQuantization::GetIntegerLengthOut(const string layer_name) {
  int pos = find(layer_names_.begin(), layer_names_.end(), layer_name)
      - layer_names_.begin();
  return il_out_[pos];
}
