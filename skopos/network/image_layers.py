from __future__ import absolute_import

import tensorflow as tf

from skopos.network.layers import Layer

class ImageLayer(Layer):
	"""docstring for ImageLayer"""
	def __init__(self):
		super(ImageLayer, self).__init__()

class Convolutional(ImageLayer):
	"""docstring for Convolutional"""
	def __init__(self, k_size=3, stride=1, filters=16, 
		padding='SAME', bias=False, weight_stddev=0.1, weight_decay=0.00004):
		super(Convolutional, self).__init__()
		self.k_size = k_size
		self.stride = stride
		self.filters = filters
		self.padding = padding
		self.bias = bias
		self.weight_stddev = weight_stddev
		self.weight_decay = weight_decay

	def apply_layer(self, out):
		
		filters_in = out.get_shape()[-1]
		shape = [self.k_size, self.k_size, filters_in, self.filters]
		initializer = tf.truncated_normal_initializer(stddev=self.weight_stddev)
		weights = Layer.get_variables('conv_weights',
	                            shape=shape,
	                            weight_decay=self.weight_decay,
	                            initializer=initializer)
		if self.bias == True:
			bias = Layer.get_variables('conv_bais',
		   							shape=[filters],
		   							initializer=initializer)
		out = tf.nn.conv2d(out, weights, [1, self.stride, self.stride, 1], self.padding)
		return out

	def get_scope(self):
		return "conv"

class MaxPool(ImageLayer):
	"""docstring for MaxPool"""
	def __init__(self, k_size=3, stride=2, padding='SAME'):
		super(MaxPool, self).__init__()
		self.k_size = k_size
		self.stride = stride
		self.padding = padding

	def apply_layer(self, out):
		out = tf.nn.max_pool(out,
	                          ksize=[1, self.k_size, self.k_size, 1],
	                          strides=[1, self.stride, self.stride, 1],
	                          padding='SAME')
		return out

	def get_scope(self):
		return "max_pool"
		