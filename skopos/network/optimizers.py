from __future__ import absolute_import
import tensorflow as tf

class Optimizer(object):
	"""docstring for Optimizer"""
	def __init__(self):
		super(Optimizer, self).__init__()

class Adam(Optimizer):
	"""docstring for Adam"""
	def __init__(self, learning_rate):
		super(Adam, self).__init__()
		self.learning_rate = learning_rate

	def initialize_optimizer(self):
		return tf.train.AdamOptimizer(learning_rate = self.learning_rate)

class Gradient(Optimizer):
	"""docstring for Adam"""
	def __init__(self, learning_rate):
		super(Gradient, self).__init__()
		self.learning_rate = learning_rate

	def initialize_optimizer(self):
		return tf.train.GradientDescentOptimizer(learning_rate = self.learning_rate)

class RMSProp(Optimizer):
	"""docstring for Adam"""
	def __init__(self, learning_rate):
		super(RMSProp, self).__init__()
		self.learning_rate = learning_rate

	def initialize_optimizer(self):
		return tf.train.RMSPropOptimizer(learning_rate = self.learning_rate)


		
	