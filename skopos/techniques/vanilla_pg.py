from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from skopos.network.simple_layers import FullyConnected
from skopos.utils.preprocessing import Preprocessing
from skopos.techniques.learners import PolicyGradientLearner

class VanillaPolicyGradient(PolicyGradientLearner):
	"""docstring for VanillaPolicyGradient"""
	def __init__(self):
		super(VanillaPolicyGradient, self).__init__()

	def define_placeholders(self):
		x = tf.placeholder(np.float32, shape=[None, self.network.get_state_dimension()])
		y_ = tf.placeholder(np.float32, shape=[None])
		actions = tf.placeholder(shape=[None], dtype=tf.int32)
		return x, y_, actions

	def output(self, x, reuse):
		output_layer = FullyConnected(size=self.network.get_action_number())
		with tf.variable_scope(self.network.get_scope() + 'output_layer', reuse=reuse):
			out = output_layer.apply_layer(x)
		return tf.nn.softmax(out)

	def error_function(self, y, y_, actions):
		actions_ohe = tf.one_hot(actions, self.network.get_action_number(), dtype=tf.float32)
		responsible_outputs = tf.reduce_sum(y*actions_ohe, reduction_indices=[1])
		loss = - tf.reduce_sum(tf.log(responsible_outputs)*y_)
		return loss

	def prediction(self, out):
		return np.argmax(out, 1)

	def train(self, batch_size, number_of_sequences):
		batch = self.agent.get_memory().get_batches(batch_size, number_of_sequences)
		end_states = (1 - np.asarray(batch.get_dones()).astype(int))
		discounted_rewards = self.discount(batch.get_rewards(), self.agent.get_discount_factor()) * end_states

		starting_states = np.asarray(batch.get_starting_states()).reshape(-1, self.network.get_state_dimension())
		
		grads = self.agent.get_sess().run(self.network.optimizer, feed_dict={
			self.network.y_: discounted_rewards,
			self.network.actions: batch.get_actions(),
			self.network.x: starting_states})
		return