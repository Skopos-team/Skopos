from __future__ import absolute_import
import tensorflow as tf
from skopos.network.layers import Layer
from skopos.utils.preprocessing import Preprocessing

class SimpleLayer(Layer):
	"""docstring for SimpleLayer"""
	def __init__(self):
		super(SimpleLayer, self).__init__()
		
class FullyConnected(SimpleLayer):
	"""docstring for Fully Connected"""
	def __init__(self, size=100, weight_decay=None, initializer=None, initialize_bias=False):
		super(FullyConnected, self).__init__()
		self.size = size
		self.weight_decay = weight_decay
		self.initializer = initializer
		self.initialize_bias = initialize_bias

	def apply_layer(self, x):
		""" One fully connected layer """
		input_dim = x.get_shape()[1]
		fc_w = Layer.get_variables(name='weights', 
	    	shape=[input_dim, self.size], 
	    	weight_decay=self.weight_decay,
	    	initializer=self.initializer)
		if self.initialize_bias == True:
			fc_b = Layer.get_variables(name='bias', 
				shape=[self.size],
				weight_decay=self.weight_decay,
				initializer=self.initializer)
		else: 
			fc_b = Layer.get_variables(name='bias', 
				shape=[self.size],
				weight_decay=self.weight_decay)		
		fc_h = tf.matmul(x, fc_w) + fc_b
		return fc_h

	def get_scope(self):
		return "fc"
		
class Relu(SimpleLayer):
	"""docstring for Relu"""
	def __init__(self):
		super(Relu, self).__init__()

	def apply_layer(self, x):
		return tf.nn.relu(x)

	def get_scope(self):
		return "relu"

class Dropout(SimpleLayer):
	"""docstring for Relu"""
	def __init__(self, keep_probability=0.8):
		super(Dropout, self).__init__()
		self.keep_prob = keep_probability

	def apply_layer(self, x):
		return tf.nn.dropout(x, self.keep_prob)

	def get_scope(self):
		return "dropout"

class Sigmoid(SimpleLayer):
	"""docstring for Relu"""
	def __init__(self):
		super(Sigmoid, self).__init__()

	def apply_layer(self, x):
		return tf.sigmoid(x)

	def get_scope(self):
		return "sigmoid"

class Tanh(SimpleLayer):
	"""docstring for Relu"""
	def __init__(self):
		super(Tanh, self).__init__()

	def apply_layer(self, x):
		return tf.tanh(x)

	def get_scope(self):
		return "tanh"

		