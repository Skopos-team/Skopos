from __future__ import absolute_import

import tensorflow as tf

from skopos.utils.preprocessing import Preprocessing 

class Network(object):
	"""docstring for NetworkBuilder"""
	def __init__(self, scope="main"):
		super(Network, self).__init__()
		self.layers = []
		self.env = None
		self.trainer = None
		self.learner = None
		self.scope = scope
		self.distributed_training = False

	def initialize_environment_variables(self):
		""" The possible observation spaces could be both Images or Vectors """
		self.act_dimension = self.env.action_space.n
		try:
			self.img_height = self.env.observation_space.shape[0]
			self.img_width = self.env.observation_space.shape[1]
			self.img_input_channels = self.env.observation_space.shape[2]
			
			self.state_shape = [self.img_height, self.img_width, self.img_input_channels]
			self.state_dimension = self.env.observation_space.shape[0]*self.env.observation_space.shape[1]*self.env.observation_space.shape[2]
		except IndexError:
			self.state_dimension = self.env.observation_space.shape[0]
		return

	def inference(self, x, reuse=False):
		for i, layer in enumerate(self.layers):
			x = self.check_input_layer(self.layers, i, x)
			with tf.variable_scope(self.scope + layer.get_scope() + str(i), reuse=reuse):
				x = layer.apply_layer(x)
				try:
					self.set_state_shape([x.shape[1], x.shape[2], x.shape[3]])
				except IndexError:
					pass
		x = Preprocessing.from_tensor_to_vector(x)
		x = self.learner.output(x, reuse)
		return x

	def prediction(self, out):
		return self.learner.prediction(out)

	def error(self, y, y_, actions):
		return self.learner.error_function(y, y_, actions)

	def optimize(self, loss):
		if self.distributed_training == True:
			local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.get_scope())
			gradients = tf.gradients(loss, local_vars)
			var_norms = tf.global_norm(local_vars)
			""" Why 40.0? """
			grads, grad_norms = tf.clip_by_global_norm(gradients, 40.0)
			global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "main")
			return self.trainer.apply_gradients(zip(grads, global_vars), name="optimizer")
		else:
			return self.trainer.minimize(loss, name="optimizer")

	def build_graph(self):
		self.initialize_environment_variables()
		self.x, self.y_, self.actions = self.learner.define_placeholders()
		self.out = self.inference(self.x, reuse=False)
		self.loss = self.error(self.out, self.y_, self.actions)
		self.optimizer = self.optimize(self.loss)

	""" User methods to build the initial network. """
	def add(self, layer):
		self.layers.append(layer)

	def set_optimizer(self, trainer):
		self.trainer = trainer.initialize_optimizer()

	""" Module methods to manage the network """
	def get_trainer(self):
		return self.trainer

	def set_trainer(self, trainer):
		self.trainer = trainer

	def get_layers(self):
		return self.layers

	def set_distributed_training(self, distributed_training):
		self.distributed_training = distributed_training

	def get_distributed_training(self):
		return self.distributed_training

	def set_layers(self, layers):
		self.layers = layers

	def set_learner(self, learner):
		self.learner = learner

	def get_learner(self):
		return self.learner

	def get_agent(self):
		return agent

	def set_environment(self, env):
		self.env = env

	def get_environment(self):
		return self.env

	def get_action_number(self):
		return self.act_dimension

	def get_state_dimension(self):
		return self.state_dimension

	def get_img_height(self):
		try:
			return self.img_height
		except AttributeError:
			return

	def get_img_width(self):
		try:
			return self.img_width
		except AttributeError:
			return

	def get_img_input_channels(self):
		try:
			return self.img_input_channels
		except AttributeError:
			return
	def get_state_shape(self):
		return self.state_shape

	def set_state_shape(self, shape):
		self.state_shape = shape

	def get_scope(self):
		return self.scope

	def check_input_layer(self, layers, i, out):
		if layers[i].get_scope() == "conv":
			out = Preprocessing.from_vector_to_image(out, self.get_state_shape())
			return out
		elif layers[i].get_scope() == "max_pool":
			out = Preprocessing.from_vector_to_image(out, self.get_state_shape())
			return out
		else:
			return Preprocessing.from_tensor_to_vector(out)

	









		


