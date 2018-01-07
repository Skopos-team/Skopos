from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import threading
import multiprocessing
from time import sleep
import copy
import Queue

""" Default technique """
from skopos.techniques.dqn import DQN

""" Default memory """
from skopos.memories.memory import ExperienceReplay, Episode

""" Utils for visualization and preprocessing """
from skopos.utils.visualize import Plot
from skopos.utils.net_processing import NetProcessing

""" Default exploration strategy """
from skopos.agent.exploration_strategy import EGreedy

""" Default policy """
from skopos.agent.policies import NetworkBasedPolicy
from skopos.agent.environment_preprocessing import Environment
from skopos.agent.agent import Agent
from skopos.network.network_builder import Network


class Worker(Agent):
	"""docstring for Worker"""
	def __init__(self, agent, name, env):
		super(Worker, self).__init__()
		self.name = "worker_" + str(name)
		self.env = env
		self.copy_agent(agent)
		self.saver = tf.train.Saver()
		if self.tensorboard_visualization == True:
			self.summary_writer = tf.summary.FileWriter(self.log_dir + "/train_" + str(name))
		print("%s, created!" % self.name)

	def copy_agent(self, agent):
		self.memory = type(agent.get_memory())() 
		self.env_preprocessing = Environment(self.env)
		self.sess = agent.get_sess()
		self.learner = type(agent.get_learner())()
		self.local_network = NetProcessing.copy_network(agent.get_network(), scope=self.name)
		self.local_network.set_distributed_training(True)
		self.local_network.set_learner(self.learner)
		self.learner.set_network(self.local_network)
		""" General parameters """
		self.training_info = agent.get_training_info()
		self.tensorboard_visualization = agent.get_tensorboard_visualization()
		self.save_model = agent.get_save_model()
		self.restore_model = agent.get_restore_model()
		self.model_folder = agent.get_model_folder()
		self.model_path = agent.get_model_path()
		self.save_model_frequency = agent.get_save_model_frequency()
		""" Running parameters """
		self.number_of_episodes = agent.get_number_of_episodes()
		self.max_episode_duration = agent.get_max_episode_duration()
		self.discount_factor = agent.get_discount_factor()
		self.pretrain_steps = agent.get_pretrain_steps()
		self.update_frequency = agent.get_update_frequency()
		self.batch_size = agent.get_batch_size()
		self.sequences = agent.get_sequences()
		self.processors = agent.get_processors()
		""" Policy and Exploration Strategy parameters """
		self.exploration_strategy = type(agent.get_exploration_strategy())()
		self.learner.set_agent(self)
		self.policy = type(agent.get_policy())()
		self.initialize_policy()
		self.local_network.build_graph()
		self.log_dir = agent.get_log_dir()
		return 

	def initialize_policy(self):
		self.policy.set_env(self.env)
		self.policy.set_network(self.local_network)
		self.policy.set_session(self.sess)
		self.exploration_strategy.set_policy(self.policy)
		return

	def work(self, T_queue):

		if self.name == "worker_0":
			self.restore_model_from_ckpt()
		
		T = T_queue.get()
		T_queue.put(T+1)
		self.reward_list = []
		self.episodes_length = []
		self.avg_max_exp_rewards = []
		
		print("Starting worker " + str(self.name))
		while T < self.number_of_episodes * self.processors:
			try:
				""" In case still doesnt exit yet """
				self.sess.run(NetProcessing.update_graph(from_scope="main", to_scope=self.name))
			except ValueError:
				pass
			""" Processing the state in order to feed it into the tensorflow network """
			s = self.env.reset()
			s = self.env_preprocessing.process_state(s)

			episode_reward = 0
			episode_avg_exp_rewards = 0
			done = False
			episode = Episode()
			episode_steps = 0
			
			for j in range(0, self.max_episode_duration):

				episode_steps = episode_steps + 1

				if T < self.pretrain_steps:
					""" Pretraining to collect experience """
					self.pretrain_steps = self.pretrain_steps - 1
					action = np.random.randint(0, self.env.action_space.n)
					expected_rewards = np.ones(self.env.action_space.n)
				else:
					""" Predicting function """
					action, expected_rewards = self.exploration_strategy.evaluate(s)
				
				landing_state, reward, done, info = self.env.step(action)
				landing_state = self.env_preprocessing.process_state(landing_state)
				""" Updating global counter """
				T = T_queue.get()
				T_queue.put(T+1)

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
						self.summary_writer.add_summary(self.summary, T)
						self.summary_writer.flush()
					break

			if self.name == "worker_0" and T % self.save_model_frequency == 0 and self.save_model == True:
				self.save_model_ckpt()

			self.reward_list.append(episode_reward)
			self.episodes_length.append(j)
			self.avg_max_exp_rewards.append(episode_avg_exp_rewards/j)
			self.memory.record_episode(episode)
			if T % self.update_frequency == 0 and T > self.pretrain_steps:
				self.learner.train(self.batch_size, self.sequences)

		return

class AsynchronousAgent(Agent):
	""" The init function takes in input the Tensorflow parameters """
	def __init__(self, env, network, processors=4, number_of_episodes=1000, 
		max_episode_duration=100, discount_factor=0.99, pretrain_steps=100,
		batch_size=32, update_frequency=1, sequences=1, 
		training_info=False, training_parameters=False,
		show_results=False, tensorboard_visualization=False, save_model=False,
		model_folder="./model/", model_path="./model/model", 
		save_model_frequency=50, restore_model=False, log_dir="./tensorboard",
		learner=DQN(), exploration_strategy=EGreedy(), 
		policy=NetworkBasedPolicy(), memory=ExperienceReplay()):
		super(AsynchronousAgent, self).__init__()
		""" Building the computation graph for running the algo """
		self.memory = memory
		self.network = network
		self.learner = learner
		""" Sharing references """
		self.env = env
		self.network.set_environment(self.env)
		self.network.set_learner(self.learner)
		self.network.set_distributed_training(True)
		self.learner.set_network(self.network)
		self.env_preprocessing = Environment(self.env)
		""" Running parameters """
		self.number_of_episodes = number_of_episodes
		self.max_episode_duration = max_episode_duration
		self.discount_factor = discount_factor
		self.pretrain_steps = pretrain_steps
		self.update_frequency = update_frequency
		self.batch_size = batch_size
		self.sequences = sequences
		""" Define general parameters """
		self.training_info = training_info
		self.training_parameters = training_parameters
		self.show_results = show_results
		self.tensorboard_visualization = tensorboard_visualization
		self.save_model = save_model
		self.restore_model = restore_model
		self.processors = processors
		self.model_folder = model_folder
		self.model_path = model_path
		self.save_model_frequency = save_model_frequency
		self.log_dir = log_dir
		""" Policy and Exploration Strategy parameters """
		self.exploration_strategy = exploration_strategy
		self.learner.set_agent(self)
		self.policy = policy
		self.sess = tf.Session()
		self.initialize_policy()
		

	def initialize_policy(self):
		self.policy.set_env(self.env)
		self.policy.set_network(self.network)
		self.policy.set_session(self.sess)
		self.exploration_strategy.set_policy(self.policy)
		return

	def run(self):
		if self.training_parameters == True:
			Plot.show_training_parameters(self)
		
		with tf.device("/cpu:0"):
			print("Global Network Initialization...")
			self.network.build_graph()
			self.sess.run(tf.global_variables_initializer())
			print("Global Network Initialized...")
			print("Creating the global counter...")
			T_queue = Queue.Queue()
			T_queue.put(0)
			self.workers = []
			print("Instantiating all the workers...")
			for i in range(0, self.processors):
				self.workers.append(Worker(self, i, env=copy.deepcopy(self.env)))
		
		self.coord = tf.train.Coordinator()
		self.sess.run(tf.global_variables_initializer())
		
		processes = []
		for worker in self.workers:
			worker_work = lambda: worker.work(T_queue=T_queue)
			t = threading.Thread(target=(worker_work))
			t.start()
			processes.append(t)
		self.coord.join(processes)

		if self.show_results == True:
			Plot.visualize_results(self.workers[0].reward_list, self.workers[0].episodes_length, self.workers[0].avg_max_exp_rewards)
		return















