from __future__ import absolute_import

import numpy as np
import random

class ExplorationStrategy(object):
	"""docstring for ExplorationStrategy"""
	def __init__(self):
		super(ExplorationStrategy, self).__init__()

	def get_environment(self):
		return self.policy.get_environment()

	def evaluate(self, state):
		raise NotImplementedError('Subclasses must override the method.')

	def set_policy(self, policy):
		self.policy = policy

class Greedy(ExplorationStrategy):
	"""docstring for Greedy"""
	def __init__(self):
		super(Greedy, self).__init__()

	def evaluate(self, state):
		action, expected_rewards = self.policy.evaluate(state)
		return action, expected_rewards 

class EGreedy(ExplorationStrategy):
	
	def __init__(self, epsilon=0.1):
		super(EGreedy, self).__init__()
		self.epsilon = epsilon

	def evaluate(self, state):
		action, expected_rewards = self.policy.evaluate(state)
		if random.random < self.epsilon:
			action = random.randint(0, self.env.action_space.n - 1)
			return action, expected_rewards
		else:
			return action, expected_rewards

class DecrementalEGreedy(ExplorationStrategy):
	
	def __init__(self, epsilon=0.4, discountFactor=0.99, minimumEpsilon=0.1):
		self.epsilon = epsilon
		self.discountFactor = discountFactor
		self.minimumEpsilon = minimumEpsilon

	def evaluate(self, state):
		action, expected_rewards = self.policy.evaluate(state)
		condition = random.random < self.epsilon
		self.epsilon = max(self.minimumEpsilon, self.epsilon * self.discountFactor)

		if condition:
			action = random.randint(0, self.env.action_space.n - 1)
			return action, expected_rewards
		else:
			return action, expected_rewards

class Boltzman(ExplorationStrategy):
	
	def __init__(self, epsilon=0.4, discountFactor=0.99, minimumEpsilon=0.1):
		self.epsilon = epsilon
		self.discountFactor = discountFactor
		self.minimumEpsilon = minimumEpsilon

	def evaluate(self, state):
		action, expected_rewards_ = self.policy.evaluate(state)
		expected_rewards = np.squeeze(expected_rewards_) 
		expected_rewards = expected_rewards + np.absolute(min(expected_rewards))
		boltzman_values = - expected_rewards / self.epsilon
		boltzman_values = [np.exp(value) for value in boltzman_values]
		boltzman_values_sum = sum(boltzman_values)
		probabilities = []

		for element in boltzman_values:
			prob = element / boltzman_values_sum
			probabilities.append(prob)


		probabilities = np.squeeze(probabilities)
		chosen_value = np.random.choice(expected_rewards, p=probabilities)
		self.epsilon = max(self.minimumEpsilon, self.epsilon * self.discountFactor)	

		return np.argmax(expected_rewards == chosen_value), expected_rewards_



		

