#!/usr/bin/env sh

log_file = "./examples/mnist_tests/3D_ST_CNN_R/LOG_TRAIN/train.log"

mkdir -p ./examples/mnist_tests/3D_ST_CNN_R/LOG_TRAIN/
mkdir -p ./examples/mnist_tests/3D_ST_CNN_R/models/

GLOG_log_dir="./examples/mnist_tests/3D_ST_CNN_R/LOG_TRAIN/" ./build/tools/caffe train --solver=examples/mnist_tests/3D_ST_CNN_R/solver.prototxt --gpu=2 # --gpu=all