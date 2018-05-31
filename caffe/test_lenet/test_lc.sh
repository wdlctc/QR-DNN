#!/usr/bin/env sh
set -e

./build/tools/caffe test --model=test_lenet/output.prototxt --weights=test_lenet/output.caffemodel --gpu=2 --iterations=500 $@
