from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import random

class Policy(object):
	"""docstring for Policy"""
	def __init__(self):
		super(Policy, self).__init__()

	def get_environment(self):
		return self.env

	def set_env(self, env):
		self.env = env

	def set_network(self, network):
		self.network = network

	def set_session(self, sess):
		self.sess = sess

class RandomPolicy(Policy):
	"""docstring for RandomPolicy"""
	def __init__(self):
		super(RandomPolicy, self).__init__()

	def evaluate(self, state):
		action = random.randint(0, self.env.action_space.n - 1)
		return action, np.ones(self.env.action_space.n)

class NetworkBasedPolicy(Policy):
	"""docstring for NetworkBasedPolicy"""
	def __init__(self):
		super(NetworkBasedPolicy, self).__init__()
		
	def evaluate(self, state):
		actions = self.sess.run(self.network.out, feed_dict={self.network.x : state})
		return int(self.network.get_learner().prediction(actions)), actions
		