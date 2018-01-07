from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from skopos.network.network_builder import Network

class NetProcessing(object):
	"""docstring for NetProcessing"""
	def __init__(self):
		super(NetProcessing, self).__init__()

	@staticmethod
	def copy_network(network, scope):
		new_network = Network(scope=scope)
		new_network.set_layers(network.get_layers())
		new_network.set_trainer(network.get_trainer())
		new_network.set_distributed_training(network.get_distributed_training())
		new_network.set_environment(network.get_environment())
		return new_network

	@staticmethod
	def update_graph(from_scope, to_scope):
		from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
		to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
		op_holder = []
		for from_var, to_var in zip(from_vars, to_vars):
			op_holder.append(to_var.assign(from_var))
		return op_holder