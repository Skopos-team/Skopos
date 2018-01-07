from __future__ import absolute_import

import tensorflow as tf
import numpy as np

class Preprocessing(object):
	"""docstring for Preprocessing"""
	def __init__(self, arg):
		super(Preprocessing, self).__init__()
		self.arg = arg

	@staticmethod
	def from_vector_to_image(x, shape):
		return tf.reshape(x, shape=[-1, shape[0], shape[1], shape[2]])

	@staticmethod
	def from_tensor_to_vector(x):
		return tf.contrib.layers.flatten(x)