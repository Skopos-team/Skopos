from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from skopos.memories.memory import ExperienceReplay, Episode
from skopos.utils.visualize import Plot
from skopos.agent.exploration_strategy import EGreedy
from skopos.agent.policies import NetworkBasedPolicy
from skopos.agent.environment_preprocessing import Environment
from skopos.agent.agent import Agent
from skopos.techniques.dqn import DQN

class SimpleAgent(Agent):

	def __init__(self, env, network, processors=4, number_of_episodes=1000, 
		max_episode_duration=100, discount_factor=0.99, pretrain_steps=100,
		batch_size=32, update_frequency=1, sequences=1, 
		training_info=False, show_results=False, tensorboard_visualization=False,
		training_parameters=False, 
		save_model=False, model_folder="./model/", model_path="./model/model", 
		save_model_frequency=50, restore_model=False, log_dir="./tensorboard",
		learner=DQN(), exploration_strategy=EGreedy(), 
		policy=NetworkBasedPolicy(), memory=ExperienceReplay()):
		super(SimpleAgent, self).__init__()
		""" Building the computation graph for running the algo """
		self.memory = memory
		self.network = network
		self.learner = learner
		""" Sharing references """
		self.env = env
		self.network.set_environment(self.env)
		self.network.set_learner(self.learner)
		self.learner.set_network(self.network)
		self.env_preprocessing = Environment(self.env)
		self.processors = processors
		self.initialize_tensorflow_computation(self.processors)
		""" Running parameters """
		self.number_of_episodes = number_of_episodes
		self.max_episode_duration = max_episode_duration
		self.discount_factor = discount_factor
		self.pretrain_steps = pretrain_steps
		self.update_frequency = update_frequency
		self.batch_size = batch_size
		self.sequences = sequences
		""" Policy and Exploration Strategy parameters """
		self.exploration_strategy = exploration_strategy
		self.learner.set_agent(self)
		self.policy = policy
		self.initialize_policy()
		""" Define general parameters """
		self.training_info = training_info
		self.training_parameters = training_parameters
		self.show_results = show_results
		self.tensorboard_visualization = tensorboard_visualization
		self.save_model = save_model
		self.restore_model = restore_model
		self.model_folder = model_folder
		self.model_path = model_path
		self.save_model_frequency = save_model_frequency
		self.log_dir = log_dir
		if self.tensorboard_visualization == True:
			self.summary_writer = tf.summary.FileWriter(self.log_dir + "/train")

	def initialize_policy(self):
		self.policy.set_env(self.env)
		self.policy.set_network(self.network)
		self.policy.set_session(self.sess)
		self.exploration_strategy.set_policy(self.policy)
		return

	def initialize_tensorflow_computation(self, processors):
		self.network.build_graph()
		self.config = tf.ConfigProto(intra_op_parallelism_threads=processors, inter_op_parallelism_threads=processors)
		self.sess = tf.Session(config=self.config)
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()
		return

	def run(self):

		if self.training_parameters == True:
			Plot.show_training_parameters(self)

		self.restore_model_from_ckpt()
		self.reward_list = []
		self.episodes_length = []
		self.avg_max_exp_rewards = []
		self.total_steps = 0

		for i in range(0, self.number_of_episodes):
			""" Processing the state in order to feed it into the tensorflow network """
			s = self.env.reset()
			s = self.env_preprocessing.process_state(s)

			episode_reward = 0
			episode_avg_exp_rewards = 0
			done = False
			episode = Episode()

			for j in range(0, self.max_episode_duration):
				self.total_steps = self.total_steps + 1

				if self.total_steps < self.pretrain_steps:
					""" Pretraining to collect experience """
					self.pretrain_steps = self.pretrain_steps - 1
					action = np.random.randint(0, self.env.action_space.n)
					expected_rewards = np.ones(self.env.action_space.n)
				else:
					""" Predicting function """
					action, expected_rewards = self.exploration_strategy.evaluate(s)
				
				landing_state, reward, done, info = self.env.step(action)
				landing_state = self.env_preprocessing.process_state(landing_state)

				""" Record statistics """
				episode.record_step(s, action, reward, landing_state, done, expected_rewards)
				episode_reward += reward
				episode_avg_exp_rewards += np.max(expected_rewards)
				s = landing_state

				if done == True:
					if self.training_info == True:
						Plot.visualize_training(j, episode_reward)
					if self.tensorboard_visualization == True:
						self.summary = tf.Summary()
						self.summary.value.add(tag='Reward', simple_value=float(np.mean(self.reward_list[-5:])))
						self.summary_writer.add_summary(self.summary, i)
						self.summary_writer.flush()
					break

			if i % self.save_model_frequency == 0 and self.save_model == True:
				self.save_model_ckpt()

			self.reward_list.append(episode_reward)
			self.episodes_length.append(j)
			self.avg_max_exp_rewards.append(episode_avg_exp_rewards/j)
			self.memory.record_episode(episode)

			if i % self.update_frequency == 0 and self.total_steps > self.pretrain_steps:
				self.learner.train(self.batch_size, self.sequences)

		if self.show_results == True:
			Plot.print_statistics(self.memory)
			Plot.visualize_results(self.reward_list, self.episodes_length, self.avg_max_exp_rewards)
		return self.reward_list
