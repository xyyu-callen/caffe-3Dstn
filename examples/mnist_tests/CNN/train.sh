#!/usr/bin/env sh

log_file = "./examples/mnist_tests/CNN/LOG_TRAIN/train.log"

mkdir -p ./examples/mnist_tests/CNN/LOG_TRAIN/
mkdir -p ./examples/mnist_tests/CNN/models/

GLOG_log_dir="./examples/mnist_tests/CNN/LOG_TRAIN/" ./build/tools/caffe train --solver=examples/mnist_tests/CNN/solver.prototxt --gpu=2