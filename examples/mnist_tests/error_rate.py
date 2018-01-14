#!/usr/bin/env python
# by xyyu

import sys
# sys.path.append("/data2/xyyu/prjs/caffe-3Dstn/python")
sys.path.insert(0, '/data2/xyyu/prjs/caffe-3Dstn/python')

import numpy as np
import os
import math
import lmdb
import caffe

from caffe.proto import caffe_pb2

s = os.sep
l = os.linesep

def softmax(v):
	y = [math.exp(k) for k in v]
	sum_y = math.fsum(y)
	z = [k/sum_y for k in y]
	return z

def main():

	# prediction mnist 'R'
	lmdb_file = '/data2/xyyu/datasets/Mnist/test_R'
	# CNN_R
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/CNN/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/CNN/models/CNN_iter_150000.caffemodel'
	# FCN_R
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/FCN/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/FCN/models/FCN_iter_150000.caffemodel'
	# ST_CNN_R
	model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_CNN/deploy.prototxt'
	model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_CNN/models/ST_CNN_iter_150000.caffemodel'
	# ST_FCN_R
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_FCN/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_FCN/models/ST_FCN_iter_150000.caffemodel'


	# prediction mnist 'RST'
	# lmdb_file = '/data2/xyyu/datasets/Mnist/test_RTS'
	# CNN_RST
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/CNN_RST/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/CNN_RST/models/CNN_iter_150000.caffemodel'
	# FCN_RST
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/FCN_RST/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/FCN_RST/models/FCN_RST_iter_150000.caffemodel'
	# ST_CNN_RST
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_CNN_RST/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_CNN_RST/models/ST_CNN_RST_iter_150000.caffemodel'
	# ST_FCN_RST
	# model_proto = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_FCN_RST/deploy.prototxt'
	# model_file = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_FCN_RST/models/ST_FCN_RST_iter_150000.caffemodel'

	# caffe init
	gpu_id = 2
	caffe.set_device(gpu_id)
	caffe.set_mode_gpu()

	data_layer = 'data'
	feature_layer = 'prob'
	scale = 0.00390625

	net = caffe.Net(model_proto, model_file, caffe.TEST)

	error_nums = 0

	# read digit as input
	lmdb_env = lmdb.open(lmdb_file)
	lmdb_txn = lmdb_env.begin()
	lmdb_cursor = lmdb_txn.cursor()
	datum = caffe_pb2.Datum()	
	for key, value in lmdb_cursor:
		datum.ParseFromString(value)
		label = datum.label
		data = caffe.io.datum_to_array(datum)
		data = data * scale
		# CxHxW to HxWxC in cv2
		# image = np.transpose(data, (1,2,0))
		net.blobs[data_layer].data[...] = data.reshape((1, data.shape[0], data.shape[1], data.shape[2]))
		output = net.forward()
		feature = np.transpose(output[feature_layer])
		# pred_mean = np.mean(feature, axis = 1)
		# pred = softmax(pred_mean)
		# pred = softmax(feature)
		index = np.argmax(feature)
		if index != label:
			error_nums += 1
	error_percentage = error_nums / 10000.0
	print 'error percentage is ' + str(error_percentage)

	for layer_name, param in net.params.iteritems():
		print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)
	filters = net.params['loc_reg'][1].data
	print filters

if __name__ == '__main__':
	main()