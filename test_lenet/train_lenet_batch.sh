#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=test_lenet/lenet_solver_batch.prototxt $@
