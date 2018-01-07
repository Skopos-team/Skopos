from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.signal

from skopos.network.simple_layers import FullyConnected
from skopos.utils.preprocessing import Preprocessing
from skopos.utils.net_processing import NetProcessing

class Learner(object):
	""" A learner is a Technique used to predict and learn the best action. """
	def __init__(self):
		super(Learner, self).__init__()
	
	def set_network(self, network):
		self.network = network

	def set_agent(self, agent):
		self.agent = agent

	def discount(self, x, gamma):
		return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1] 

class ValueIterationLearner(Learner):
	"""docstring for ClassName"""
	def __init__(self):
		super(ValueIterationLearner, self).__init__()

	def initialize_double_network():
		target_network = NetProcessing.copy_network(self.network, "target")
		target_network.build_graph()
		target_network.set_learner(self)
		return target_network

	def run_double_network(self):
		if self.double_network is not None and self.double_network.get_network() is not None:
			self.double_network.set_network(self.initialize_double_network()) 
		""" Updates the variables of the target network using the variables of the main one """
		if self.double_network.get_total_updates() % self.double_network.get_tau() == 0:
			self.agent.get_sess().run(NetProcessing.update_graph(self.network.get_scope(), to_scope=self.double_network.get_network().get_scope()))
		final_states = np.asarray(batch.get_final_states()).reshape(-1, self.network.get_state_dimension())
		double_q = self.agent.get_sess().run(
			self.target_network.out, 
			feed_dict={self.target_network.x: final_states})
		max_double_q = double_q[range(batch_size), actions]
		target_q = batch.get_rewards() + self.agent.get_discount_factor() * max_double_q * end_states
		return target_q

class PolicyGradientLearner(Learner):
	"""docstring for ClassName"""
	def __init__(self):
		super(PolicyGradientLearner, self).__init__()

class ValuePolicyLearner(Learner):
	"""docstring for ClassName"""
	def __init__(self):
		super(ValuePolicyLearner, self).__init__()


		
		
