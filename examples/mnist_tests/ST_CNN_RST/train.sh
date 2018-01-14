#!/usr/bin/env sh

log_file = "./examples/mnist_tests/ST_CNN_RST/LOG_TRAIN/train.log"

mkdir -p ./examples/mnist_tests/ST_CNN_RST/LOG_TRAIN/
mkdir -p ./examples/mnist_tests/ST_CNN_RST/models/

GLOG_log_dir="./examples/mnist_tests/ST_CNN_RST/LOG_TRAIN/" ./build/tools/caffe train --solver=examples/mnist_tests/ST_CNN_RST/solver.prototxt --gpu=2