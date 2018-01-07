from __future__ import absolute_import

import numpy as np

class Environment(object):
	"""docstring for EnvironmentPreprocessing"""
	def __init__(self, env):
		super(Environment, self).__init__()
		self.env = env

	def process_state(self, state):
		try:
			processed_state = np.zeros((1, self.env.observation_space.n))
			processed_state[0][state] = 1
		except AttributeError:
			processed_state = np.reshape(state, (-1, np.asarray(state).flatten().shape[0]))
		return processed_state
		