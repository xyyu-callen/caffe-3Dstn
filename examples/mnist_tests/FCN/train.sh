#!/usr/bin/env sh

log_file = "./examples/mnist_tests/FCN/LOG_TRAIN/train.log"

mkdir -p ./examples/mnist_tests/FCN/LOG_TRAIN/
mkdir -p ./examples/mnist_tests/FCN/models/

GLOG_log_dir="./examples/mnist_tests/FCN/LOG_TRAIN/" ./build/tools/caffe train --solver=examples/mnist_tests/FCN/solver.prototxt --gpu=2