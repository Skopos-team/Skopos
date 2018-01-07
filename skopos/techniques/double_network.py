from __future__ import absolute_import

import tensorflow as tf
import numpy as np

class DoubleNetwork(object):
	"""docstring for DoubleNetwork"""
	def __init__(self, tau=5):
		super(DoubleNetwork, self).__init__()
		self.tau = tau
		self.total_updates = 0

	def set_network(self, network):
		self.network = network

	def get_network(self):
		return self.network

	def increment_total_updates(self):
		self.total_updates += 1

	def get_total_updates(self):
		return self.total_updates
