from __future__ import absolute_import

""" Importing the Learner """
from skopos.techniques.dqn import DQN
from skopos.techniques.dueling_dqn import Dueling
from skopos.techniques.double_network import DoubleNetwork
from skopos.techniques.vanilla_pg import VanillaPolicyGradient
from skopos.techniques.actor_critic import AdvantageActorCritic

""" Modules for building the network """
from skopos.network.network_builder import Network

""" Layers to use """
from skopos.network.simple_layers import FullyConnected, Relu, Sigmoid, Tanh
from skopos.network.image_layers import Convolutional

""" Chosen optimizer """
from skopos.network.optimizers import Adam, Gradient

""" Chosen agent"""
from skopos.agent.simple_agent import SimpleAgent
from skopos.agent.asynch_agent import AsynchronousAgent

""" Chosen expl. strategy """
from skopos.agent.exploration_strategy import EGreedy, DecrementalEGreedy

""" Chosen memory """
from skopos.memories.memory import ExperienceReplay

""" Chosen policy """
from skopos.agent.policies import NetworkBasedPolicy
from skopos.agent.policies import RandomPolicy

""" Importing OpenAI Gym to test the methods """
import gym

def main():

	""" Testing environments """
	# env_image = gym.make('Asterix-v0') # Box Image
	# env_discrete = gym.make('FrozenLake-v0') # Discrete 
	env_box = gym.make('CartPole-v0') # Box
	# env = gym.make('Roulette-v0') 

	""" Definition of the Network """
	network = Network()
	network.add(FullyConnected(size=24))
	network.add(Sigmoid())
	network.add(FullyConnected(size=24))
	network.add(Sigmoid())
	network.set_optimizer(Gradient(learning_rate=0.4))

	""" Definition of the learner, exploration strategy and policy """
	exploration_strategy = DecrementalEGreedy(epsilon=1)
	policy = NetworkBasedPolicy()
	memory = ExperienceReplay()
	learner = DQN()
	
	""" Definition of the Agent """
	agent = SimpleAgent(
		
		env=env_box, 
		network=network, 
		
		processors=4, 
		number_of_episodes=1000, 
		max_episode_duration=1000,
		discount_factor=0.9, 
		pretrain_steps=100,
		batch_size=64,
		update_frequency=1,
		sequences=1,   
		training_info=True,
		show_results=True,
		tensorboard_visualization=False,
		training_parameters=False,
		
		save_model=False,
		restore_model=False,

		learner=learner,
		exploration_strategy=exploration_strategy, 
		policy=policy,
		memory=memory
		)
        
	reward_list = agent.run()
	return

if __name__ == '__main__':
    main()


