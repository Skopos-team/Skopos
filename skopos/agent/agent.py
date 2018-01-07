from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from skopos.memories.memory import ExperienceReplay, Episode
from skopos.utils.visualize import Plot
from skopos.agent.exploration_strategy import EGreedy
from skopos.agent.policies import NetworkBasedPolicy
from skopos.agent.environment_preprocessing import Environment

class Agent(object):
	"""docstring for Agent"""
	def __init__(self):
		super(Agent, self).__init__()

	def get_sess(self):
		return self.sess

	def get_coord(self):
		return self.coord

	def get_environment(self):
		return self.env

	def get_network(self):
		return self.network

	def get_processors(self):
		return self.processors

	def get_number_of_episodes(self):
		return self.number_of_episodes

	def get_max_episode_duration(self):
		return self.max_episode_duration

	def get_discount_factor(self):
		return self.discount_factor

	def get_pretrain_steps(self):
		return self.pretrain_steps

	def get_update_frequency(self):
		return self.update_frequency

	def get_batch_size(self):
		return self.batch_size

	def get_sequences(self):
		return self.sequences

	def get_model_folder(self):
		return self.model_folder

	def get_learner(self):
		return self.learner

	def get_memory(self):
		return self.memory

	def get_exploration_strategy(self):
		return self.exploration_strategy

	def get_policy(self):
		return self.policy

	def get_training_info(self):
		return self.training_info

	def get_tensorboard_visualization(self):
		return self.tensorboard_visualization

	def get_save_model(self):
		return self.save_model

	def get_restore_model(self):
		return self.restore_model

	def get_model_path(self):
		return self.model_path

	def get_save_model_frequency(self):
		return self.save_model_frequency

	def get_log_dir(self):
		return self.log_dir

	def restore_model_from_ckpt(self):
		if self.restore_model == True:
			self.saver = tf.train.import_meta_graph(self.model_path + '.meta')
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_folder))
			print("Model restored from checkpoint")
		return

	def save_model_ckpt(self):
		model_path = self.saver.save(self.sess, self.model_path)
		print("Model saved in: ", model_path)
		return



		
		

		

