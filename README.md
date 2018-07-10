# Quantization and Reconstruction algorithm for Quantized and Reconstructed Deep Neural Networks

Code example for the FPL 2018 short paper. We modify the original [caffe](http://caffe.berkeleyvision.org/) to implement our QR-DNN algorithm. And we also thank the example code provied by [Ristretto](http://ristretto.lepsucd.com/).

## Prerequisites
- NVIDIA GPU + CUDA + CuDNN
- python2.7
- Imagenet dataset


## QR-DNN usage

### Command to train a fixed-point AlexNet with 8-bit actiavtions and weight,
`python python/merge_all.py --model=test/train_val.prototxt  --mode=dynamic_fixed_point --solver_first=test/train_alex.solver`

### Command to train a logrithmic AlexNet with 8-bit actiavtions and 4-bit weight,
`python python/merge_all.py --model=test/train_val.prototxt  --mode=Power-of-two --solver_first=test/train_alex.solver --solver_second=test/train_alex_second.solver`
