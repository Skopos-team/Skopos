from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from skopos.network.simple_layers import FullyConnected
from skopos.utils.preprocessing import Preprocessing
from skopos.techniques.learners import PolicyGradientLearner

class AdvantageActorCritic(PolicyGradientLearner):

	def __init__(self, value_reg=0.5, entropy_reg=0.01):
		super(AdvantageActorCritic, self).__init__()
		self.value_reg = value_reg
		self.entropy_reg = entropy_reg

	def define_placeholders(self):
		x = tf.placeholder(tf.float32, shape=[None, self.network.get_state_dimension()])
		y_ = tf.placeholder(tf.float32, shape=[None])
		actions = tf.placeholder(shape=[None], dtype=tf.int32)
		return x, y_, actions

	def output(self, x, reuse):
		policy_layer = FullyConnected(size=self.network.get_action_number(), initializer=self.columns_initializer(0.01))
		with tf.variable_scope(self.network.get_scope() + 'policy_layer', reuse=reuse):
			policy = policy_layer.apply_layer(x)
			policy = tf.nn.softmax(policy)

		value_layer = FullyConnected(size=1)
		with tf.variable_scope(self.network.get_scope() + 'value_layer', reuse=reuse, initializer=self.columns_initializer(1.0)):
			self.value = value_layer.apply_layer(x)
		return policy

	def prediction(self, out):
		out = out.flatten()
		probabilities = out.clip(min=0)
		probabilities /= probabilities.sum()
		action = np.random.choice(out, p=probabilities)
		action = np.argmax(out == action)
		return action

	def error_function(self, y, y_, actions):
		""" Renaming the variables to better understand """
		policy = y
		value = self.value
		discounted_rewards = y_

		actions_ohe = tf.one_hot(actions, self.network.get_action_number(), dtype=tf.float32)
		
		##### Policy loss ###### 
		""" We add a small constant to prevent a NaN error that could occur 
		if we selected an action while it is probability was zero. """
		logp = tf.log(tf.reduce_sum(policy * actions_ohe, [1]) + 1e-10)
		advantages = discounted_rewards - value
		policy_loss = - logp * tf.stop_gradient(advantages)

		##### Value loss ######
		value_loss = self.value_reg * tf.square(advantages)

		##### Entropy #####
		entropy = - self.entropy_reg * tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1, keep_dims=True)

		return tf.reduce_mean(value_loss + policy_loss + entropy)

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

	def columns_initializer(self, std=0.1):
		def _initializer(shape, dtype=None, partition_info=None):
			out = np.random.randn(*shape).astype(np.float32)
			out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
			return tf.constant(out)
		return _initializer
