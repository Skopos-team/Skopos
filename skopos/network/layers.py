from __future__ import absolute_import

import tensorflow as tf

class Layer(object):
	"""docstring for Layers"""
	def __init__(self):
		super(Layer, self).__init__()

	@staticmethod
	def get_variables(name, shape, weight_decay=None, initializer=tf.contrib.layers.xavier_initializer()):
		""" Getting a variable """
		if weight_decay is not None:
			regularizer = tf.contrib.layers.l2_regularizer(scale=weight_decay)
			new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)
		else:
			new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
		return new_variables

	def set_network(self, network):
		self.network = network

		
