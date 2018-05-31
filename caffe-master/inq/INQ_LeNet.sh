#!/usr/bin/env sh

./build/tools/inq quantize \
	--model=test_lenet/final.prototxt \
	--weights=test_lenet/final.caffemodel \
	--model_quantized=test_lenet/quantized.prototxt \
	--solver=test_lenet/solver.prototxt \
	--trimming_mode=sb\
	--gpu=2 --iterations=100 
