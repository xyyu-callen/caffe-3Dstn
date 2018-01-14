#!/usr/bin/env sh
mkdir -p ./examples/tristnp3d/LOG_TRAIN/
mkdir -p ./examples/tristnp3d/models/
GLOG_log_dir="./examples/tristnp3d/LOG_TRAIN/" ./build/tools/caffe train --solver=examples/tristnp3d/solver.prototxt --gpu all
#./build/tools/caffe train --solver=examples/tridim_frgc/solver.prototxt --gpu all
