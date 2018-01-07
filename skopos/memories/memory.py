# coding=utf-8

from __future__ import absolute_import
import random
import numpy as np


class Memory(object):
	"""Superclass including the basic and static methods for the memory utilties"""
	def __init__(self):
		super(Memory, self).__init__()

	""" Methods to extract the index of the list to be replaced"""
	def random_replace(number_of_episodes):
		""" Just returns a random number within list size """
		return random.randint(0, number_of_episodes)

	def fifo_replace(number_of_episodes):
		""" Return the index of the oldest element in the list """
		return 0

	def lifo_replace(number_of_episodes):
		""" Return the index of the latest element added in the list """
		return number_of_episodes - 1

	""" Methods to compute the priority level of an episode """
	def constant_priority(episode):
		""" Assigns constant priority to all the episodes (no prioritization) """
		return 1.0

	def tot_reward_difference_priority(episode):
		""" Assigns as priority the sum of the difference between expected and obtained 
			rewards at each step of the episode """
		diff = 0
		for index in range(0, episode.length()):
			choice = episode.action_sequences[index]
			diff += np.absolute(episode.reward_sequences[index] - np.squeeze(episode.expected_rewards[index])[choice])

		return diff / episode.length()

	def max_reward_difference_priority(episode):
		""" Assigns as priority the max difference between expected and obtained 
			rewards of the episode """
		diff = 0
		for index in range(0, episode.length()):
			choice = episode.action_sequences[index]
			diff = max(np.absolute(episode.reward_sequences[index] - episode.expected_rewards[index][choice]), diff)

		return diff

	def undecided_cases_priority(episode):
		""" Measures the indecision of the outuput of the policy by checking
			wether the second optimal choice is close in value to the first, the third 
			to the second and so on and so forth """
		var = 0
		for index in range(0, episode.length):
			ex_rew = list(episode.expected_rewards[index])
			length = 0
			ex_rew = ex_rew.sort()
			for i in range(1, length):
				if ex_rew[i] != 0:
					var += (ex_rew[i-1] / ex_rew[i]) * i
				length += i

			if length != 0:
				var = var / length

		return var

	# Dictionary mapping names of replacement strategies to the respective function
	replacement_strategies = {'random' : random_replace, 'fifo' : fifo_replace, 'lifo' : lifo_replace}
	# Dictionary mapping names of prioritization strategies to the respective function
	prioritization_strategies = {'constant' : constant_priority, 'tot_td' : tot_reward_difference_priority, 'max_td' : max_reward_difference_priority, 'indecision' : undecided_cases_priority}


class ExperienceReplay(Memory):
	""" Class to store the experience """
	def __init__(self, capacity=-1, replacement_strategy='random'):
		super(ExperienceReplay, self).__init__()
		self.episodes = []
		self.episode_number = 0
		self.total_steps = 0
		self.average_total_reward = 0
		self.capacity = capacity
		self.replacement_strategy = Memory.replacement_strategies[replacement_strategy]
		self.priorities = []
		self.prioritization_strategy = Memory.prioritization_strategies['constant']

	def record_episode(self, episode):
		if self.capacity == self.total_steps:
			to_be_replaced = replacement_strategy(self.episode_number)
			self.episodes.pop(to_be_replaced)
			self.priorities.pop(to_be_replaced)
		self.episodes.append(episode)
		self.episode_number += 1
		self.total_steps += episode.length()
		rew_sum = sum(episode.reward_sequences)
		self.average_total_reward += (rew_sum - self.average_total_reward) / self.episode_number
		self.priorities.append(self.prioritization_strategy(episode))
		if(self.episode_number==1):
			self.record_episode(episode)

	""" I need a method get_last_episode """
	def get_episode(self, index):
		if index > self.episode_number:
			print("Index too high. Totally available {} episodes.".format(episode_number))
			return
		return self.episodes[index]

	def get_random_episodes(self, number, prioritized=True):
		if number > self.episode_number:
			print("There are not {} episodes. Only {} available.".format(number, episode_number))
			return

		if prioritized :
			episodes = np.random.choice(self.episodes, size=number, p=(self.priorities / sum(self.priorities)))
		else:
			episodes = np.random.choice(self.episodes, size=number)

		return episodes

	def get_sequences(self, number, size, fixed_length=True, prioritized=True):
		sequences = []

		if prioritized :
			priorities = np.asarray(self.priorities) / sum(self.priorities)
		else:
			priorities = None

		priorities = np.squeeze(priorities)
		while(len(sequences) < number):
			episode = np.random.choice(self.episodes, p=priorities)
			seq = episode.get_sequences(size, fixed_length)
			if seq is not None:
				sequences.append(seq)

		return sequences

	def get_batch(self, size):
		batch = StepBatch()
		for i in range(0, size):
			episode = random.randint(0, self.episode_number)
			""" HERE AN ERROR, in case of the extracted random number is 0, it gives index out of range 

			MOREOVER IF THE GAME IS LONG (BEFORE LOOSING OR WIN TAKES A LOT OF TIME, WE DON'T HAVE ANY 
			AT THE MOMENT OF THE FIRST UPDATE 

			"""
			step = random.randint(0, self.episodes[episode-1].length())
			batch.record_step(self.episodes[episode-1].get_step(step-1))
		return batch

	def get_batches(self, number, size, sequential=True, prioritized=True):
		batches = SequenceBatch()

		if prioritized :
			priorities = np.asarray(np.squeeze(self.priorities)) / sum(np.squeeze(self.priorities))
		else:
			priorities = None

		episodes = []

		priorities = np.squeeze(priorities)
		while(batches.length() < number):
			if episodes == []:
				episodes = list(np.random.choice(self.episodes, number, p=priorities))

			episode = episodes.pop(0)
			

			if sequential:
				sequence = episode.get_sequence(size)
			else:
				sequence = episode.get_batch(size)

			if sequence is not None:
				batch = StepBatch(sequential=sequential)
				for step in sequence:
					batch.record_step(step)

				batches.record_sequence(batch)

		return batches

	def get_statistics(self):
		steps_per_episode = float(self.total_steps / self.episode_number)
		reward_per_step = self.average_total_reward / steps_per_episode
		return self.episode_number, self.total_steps, reward_per_step, self.average_total_reward



class Episode:

	def __init__(self):
		self.action_sequences = []
		self.reward_sequences = []
		self.landing_states = []
		self.starting_states = []
		self.dones = []
		self.expected_rewards = []
		self.episode_length = 0 

	def get_final_states(self):
		return np.squeeze(self.landing_states)

	def get_rewards(self):
		return np.squeeze(self.reward_sequences)

	def get_starting_states(self):
		return np.squeeze(self.starting_states)

	def get_actions(self):
		return np.squeeze(self.action_sequences)

	def length(self):
		return np.squeeze(self.episode_length)

	def get_batch(self, size, fixed_length=True):
		size = min(size, episode_length)
		indices = random.sample(range(0, episode_length), size)
		batch = [self.get_step(index) for index in indices]

		if len(batch) != size and fixed_length:
			return None

		return batch

	def get_dones(self):
		return np.squeeze(self.dones)

	def get_sequence(self, size, fixed_length=True):
		if self.episode_length >= size:
			starting_index = random.randint(0, self.episode_length-size)
		else:
			if fixed_length:
				return None
			starting_index = random.randint(0, self.episode_length)

		ending_index = min(starting_index+size, self.episode_length)
		sample = [self.get_step(index) for index in range(starting_index, ending_index)]

		return sample

	def get_step(self, index):
		return self.starting_states[index], self.action_sequences[index], self.reward_sequences[index], self.landing_states[index], self.dones[index]

	def record_step(self, starting_state, action, reward, final_state, done, expected_rewards):
		self.action_sequences.append(action)
		self.reward_sequences.append(reward)
		self.landing_states.append(final_state)
		self.starting_states.append(starting_state)
		self.dones.append(done)
		self.expected_rewards.append(expected_rewards)
		self.episode_length += 1
		return


class StepBatch:

	def __init__(self, sequential=False):
		self.steps = []
		self.tot_steps = 0
		self.sequential = sequential

	def is_sequential(self):
		return self.sequential

	def record_step(self, step):
		self.steps.append(step)
		self.tot_steps += 1

	def get_rewards(self):
		return np.squeeze([step[2] for step in self.steps])

	def get_starting_states(self):
		return np.squeeze([step[0] for step in self.steps])

	def get_final_states(self):
		return np.squeeze([step[3] for step in self.steps])

	def get_actions(self):
		return np.squeeze([step[1] for step in self.steps])

	def get_dones(self):
		return np.squeeze([step[4] for step in self.steps])

	def toList(self):
		return self.steps


class SequenceBatch:

	def __init__(self):
		self.sequences = list()
		self.tot_sequences = 0

	def length(self):
		return self.tot_sequences

	def record_sequence(self, step_batch):
		self.sequences.append(step_batch)
		self.tot_sequences += 1

	def get_rewards(self):
		rewards = [0]
		for sequence in self.sequences:
			try:
				_list = sequence.get_rewards()
				_to_add = list(_list)

			except Exception as e:
				_to_add = []
				_to_add.append(_list)
				
			rewards.append(np.asarray(_to_add))
		
		rewards.pop(0)
		rewards = np.asarray(rewards).flatten()
		return rewards

	def get_starting_states(self):
		starting_states = [0]
		for sequence in self.sequences:
			starting_states.extend(sequence.get_starting_states())		
		starting_states.pop(0)
		return starting_states

	def get_final_states(self):
		final_states = [0]
		for sequence in self.sequences:
			final_states.extend(sequence.get_starting_states())		
		final_states.pop(0)
		return final_states

	def get_actions(self):
		actions = [0]
		for sequence in self.sequences:
			try:
				_list = sequence.get_actions()
				_to_add = list(_list)

			except Exception as e:
				_to_add = []
				_to_add.append(_list)
				
			actions.append(np.asarray(_to_add))
		
		actions.pop(0)
		actions = np.asarray(actions).flatten()
		return actions

	def get_dones(self):
		dones = [0]
		for sequence in self.sequences:
			try:
				_list = sequence.get_dones()
				_to_add = list(_list)

			except Exception as e:
				_to_add = []
				_to_add.append(_list)

			dones.append(np.asarray(_to_add))
		
		dones.pop(0)
		dones = np.asarray(dones).flatten()
		return dones

	def toList(self):
		return self.sequences


class PrioritizedExperienceReplay(ExperienceReplay):
	"""docstring for Memory"""
	def __init__(self, capacity=-1, replacement_strategy='random', prioritization='tot_td'):
		super(ExperienceReplay, self).__init__()
		self.episodes = []
		self.episode_number = 0
		self.total_steps = 0
		self.average_total_reward = 0
		self.capacity = capacity
		self.replacement_strategy = Memory.replacement_strategies[replacement_strategy]
		self.priorities = []
		self.prioritization_strategy = Memory.prioritization_strategies[prioritization]

	def record_episode(self, episode):
		if self.capacity == self.total_steps:
			to_be_replaced = replacement_strategy(self.episode_number)
			self.episodes.pop(to_be_replaced)
			self.priorities.pop(to_be_replaced)
		self.episodes.append(episode)
		self.episode_number += 1
		self.total_steps += episode.length()
		rew_sum = sum(episode.reward_sequences)
		self.average_total_reward += (rew_sum - self.average_total_reward) / self.episode_number
		self.priorities.append(self.prioritization_strategy(episode))
		if(self.episode_number==1):
			self.record_episode(episode)

	""" I need a method get_last_episode """
	def get_episode(self, index):
		if index > self.episode_number:
			print("Index too high. Totally available {} episodes.".format(episode_number))
			return
		return self.episodes[index]

	def get_random_episodes(self, number, prioritized=True):
		if number > self.episode_number:
			print("There are not {} episodes. Only {} available.".format(number, episode_number))
			return

		if prioritized :
			episodes = np.random.choice(self.episodes, size=number, p=(self.priorities / sum(self.priorities)))
		else:
			episodes = np.random.choice(self.episodes, size=number)

		return episodes

	def get_sequences(self, number, size, fixed_length=True, prioritized=True):
		sequences = []

		if prioritized :
			priorities = np.asarray(self.priorities) / sum(self.priorities)
		else:
			priorities = None

		priorities = np.squeeze(priorities)
		while(len(sequences) < number):
			episode = np.random.choice(self.episodes, p=priorities)
			seq = episode.get_sequences(size, fixed_length)
			if seq is not None:
				sequences.append(seq)

		return sequences

	def get_batch(self, size):
		batch = StepBatch()
		for i in range(0, size):
			episode = random.randint(0, self.episode_number)
			""" HERE AN ERROR, in case of the extracted random number is 0, it gives index out of range 

			MOREOVER IF THE GAME IS LONG (BEFORE LOOSING OR WIN TAKES A LOT OF TIME, WE DON'T HAVE ANY 
			AT THE MOMENT OF THE FIRST UPDATE 

			"""
			step = random.randint(0, self.episodes[episode-1].length())
			batch.record_step(self.episodes[episode-1].get_step(step-1))
		return batch

	def get_batches(self, number, size, sequential=True, prioritized=True):
		batches = SequenceBatch()

		if prioritized :
			priorities = np.asarray(np.squeeze(self.priorities)) / sum(np.squeeze(self.priorities))
		else:
			priorities = None

		episodes = []

		priorities = np.squeeze(priorities)
		while(batches.length() < number):
			if episodes == []:
				episodes = list(np.random.choice(self.episodes, number, p=priorities))

			episode = episodes.pop(0)
			

			if sequential:
				sequence = episode.get_sequence(size)
			else:
				sequence = episode.get_batch(size)

			if sequence is not None:
				batch = StepBatch(sequential=sequential)
				for step in sequence:
					batch.record_step(step)

				batches.record_sequence(batch)

		return batches

	def get_statistics(self):
		steps_per_episode = float(self.total_steps / self.episode_number)
		reward_per_step = self.average_total_reward / steps_per_episode
		return self.episode_number, self.total_steps, reward_per_step, self.average_total_reward
