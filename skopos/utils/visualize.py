from __future__ import absolute_import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class Plot(object):
	"""docstring for Plot"""
	def __init__(self, arg):
		super(Plot, self).__init__()
		self.arg = arg

	@staticmethod
	def visualize_results(reward_list, episodes_length, avg_max_exp_rewards):
		gs = gridspec.GridSpec(1, 3
			)
		plt.figure(1, figsize=(12, 6))
		ax = plt.subplot(gs[0, 0])
		ax.set_title("Rewards per episode")
		ax.plot(reward_list)

		ax = plt.subplot(gs[0, 1])
		ax.set_title("Number of actions (length) per episode")
		ax.plot(episodes_length)

		ax = plt.subplot(gs[0, 2])
		ax.set_title("Avg expected reward per episode")
		ax.plot(avg_max_exp_rewards)

		plt.show()
		return

	@staticmethod
	def visualize_training(step, episode_reward):
		print("Game finished after %s iterations. With an avg reward of %s." %(step, episode_reward/step))
		return

	@staticmethod
	def print_statistics(experience):
		a, b, c, d = experience.get_statistics()
		print("{} played games for a total of {} action taken. In average I score {} per action for a total average reward per episode of {}." .format(a,b,c,d))

	@staticmethod
	def show_training_parameters(agent):
		print("Used processors during the training: ", agent.get_processors())
		print("Number of episodes: ", agent.get_number_of_episodes())
		print("Max episode duration: ", agent.get_max_episode_duration())
		print("Discount factor (lambda for value iteration learners and dropping rate for computing discounted rewards): ", agent.get_discount_factor())
		print("Number of Pre Train steps done randomly to collect a certain experience: ", agent.get_pretrain_steps())
		print("Frequency update (how often compute an optimization step): ", agent.get_update_frequency())
		
		print("Batch size (number of episodes): ", agent.get_batch_size())
		print("Number of consecutive experiences in the episode to consider (RNN, Eligibity trace): ", agent.get_sequences())
		
		print("Save the model? ", agent.get_save_model())
		print("Folder path where to save the model ckpt: ", agent.get_model_folder())
		print("Restore the model? ", agent.get_restore_model())
		print("Model path where to restore the model: ", agent.get_model_path())
		print("How often save the model: ", agent.get_save_model_frequency())
		print("Tensorboard directory: ", agent.get_log_dir())
		
		print("Learner used for training: ", agent.get_learner())
		print("Memory used for storing experiences: ", agent.get_memory())
		print("Exploration strategy used for training: ", agent.get_exploration_strategy())
		print("Policy used for training: ", agent.get_policy())
