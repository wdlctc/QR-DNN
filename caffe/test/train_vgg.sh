#!/usr/bin/env sh

./build/tools/caffe train --solver=test/train_vgg.solver -gpu=0
