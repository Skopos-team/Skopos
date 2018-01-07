from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from skopos.network.simple_layers import FullyConnected
from skopos.utils.preprocessing import Preprocessing
from skopos.techniques.learners import ValueIterationLearner

class DQN(ValueIterationLearner):

	def __init__(self, double_network=None):
		super(DQN, self).__init__()
		self.double_network = double_network

	def define_placeholders(self):
		x = tf.placeholder(tf.float32, shape=[None, self.network.get_state_dimension()])
		y_ = tf.placeholder(tf.float32, shape=[None])
		actions = tf.placeholder(tf.int32, shape=[None])
		return x, y_, actions

	def output(self, x, reuse):
		layer = FullyConnected(size=self.network.get_action_number())
		with tf.variable_scope(self.network.get_scope() + "output_layer", reuse=reuse):
			out = layer.apply_layer(x)
		return out

	def prediction(self, out):
		return np.argmax(out, 1)

	def error_function(self, y, y_, actions):
		actions_ohe = tf.one_hot(actions, self.network.get_action_number(), dtype=tf.float32)
		q = tf.reduce_sum(y * actions_ohe, reduction_indices=1)
		return tf.reduce_mean(tf.square(y_ - q), name="loss")

	def train(self, batch_size, number_of_sequences):
		batch = self.agent.get_memory().get_batches(batch_size, number_of_sequences)
		final_states = np.asarray(batch.get_final_states()).reshape(-1, self.network.get_state_dimension())
		next_step_rewards = self.agent.get_sess().run(
			self.network.out, 
			feed_dict={self.network.x: final_states})
		actions = np.argmax(next_step_rewards, 1)
		end_states = (1 - np.asarray(batch.get_dones()).astype(int))

		try:
			target_q = self.run_double_network()
		except AttributeError:
			max_expected_reward = np.max(next_step_rewards, 1)
			target_q = batch.get_rewards() + self.agent.get_discount_factor() * max_expected_reward * end_states

		starting_states = np.asarray(batch.get_starting_states()).reshape(-1, self.network.get_state_dimension())
		self.agent.get_sess().run(
			self.network.optimizer, 
			feed_dict={self.network.x: starting_states, self.network.y_: target_q, self.network.actions: actions})
		
		return