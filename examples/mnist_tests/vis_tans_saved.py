#!/usr/bin/env python

# visualization for mnist digit transforation
# by xyyu

import sys
sys.path.insert(0, '/data2/xyyu/prjs/caffe-3Dstn/python')
import caffe
import os 
import numpy as np
import random
import math
import lmdb
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from caffe.proto import caffe_pb2


def vis_square(data, saved_path):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.axis('off')
    plt.savefig(saved_path)
    plt.show()

def main():

	model_path = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests/ST_CNN/'
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

	net = caffe.Net(model_proto, model_file, caffe.TEST)
	data_layer = 'data'
	feature_layer = 'prob'
	scale = 0.00390625
	img_path = '/data2/xyyu/datasets/Mnist/test_R_ran_img'
	img_saved = '/data2/xyyu/prjs/caffe-3Dstn/examples/mnist_tests'

	img_list = os.listdir(img_path)
	for i in img_list:
		img_dir = os.path.join(img_path, i)
		frm_ = Image.open(img_dir)
		frm = np.array(frm_, dtype=np.float32)
		data = frm * scale
		net.blobs[data_layer].data[...] = data.reshape((1, 1, data.shape[0], data.shape[1]))
		output = net.forward()
		feature = np.transpose(output[feature_layer])
		st_feature = net.blobs['st_output'].data[0]
		saved_path = os.path.join(model_path,'saved_img', i)
		vis_square(st_feature, saved_path)


if __name__ == '__main__':
	res = main()