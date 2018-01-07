# Skopos

Skopos is a flexible Deep Reinforcement Learning Library which gives the possibility to the end user to implement and test faster the different Deep Reinforcement Learning techniques. The structure is based on hierarchical blocks; you can plug different techniques and experiment different combinations of “blocks”. For example using an AsynchronousAgent and AdvantageActorCritic as learning technique you generate A3C method.

### Library main assumption

The given environment must have the same characteristics as an OpenAI Gym environment. The methods used in the library are:

- env.reset(): to restart the episode.
- env.action_space.n: to get the number of possible action.
- env.step(action): to pass from a state to another. 
- env.observation_space.shape[i]: to get the input state dimension.

### Installation

You can clone the repository or install it from pip.

	$ pip install skopos 

### Usage

```python

""" Import all the necessaries classes to define and build your agent """
from skopos.techniques.dqn import DQN
from skopos.network.network_builder import Network
from skopos.network.simple_layers import FullyConnected, Sigmoid
from skopos.network.optimizers import Gradient
from skopos.agent.simple_agent import SimpleAgent
from skopos.agent.exploration_strategy import DecrementalEGreedy
from skopos.memories.memory import ExperienceReplay
from skopos.agent.policies import NetworkBasedPolicy

""" Importing OpenAI Gym to test the methods """
import gym

""" Take an environment from OpenAI Gym """
env_box = gym.make('CartPole-v0') 

""" Define the newtwork """
network = Network()
network.add(FullyConnected(size=24))
network.add(Sigmoid())
network.add(FullyConnected(size=24))
network.add(Sigmoid())
network.set_optimizer(Gradient(learning_rate=0.1))

""" Define the exploration strategy, the type of policy, the memory and the learner """
exploration_strategy = DecrementalEGreedy(epsilon=0.9)
policy = NetworkBasedPolicy()
memory = ExperienceReplay()
learner = DQN()

""" Finally create the agent and make it run """
agent = SimpleAgent(
		
	env=env_box, 
	network=network, 
	
	processors=4, 
	number_of_episodes=1000, 
	max_episode_duration=1000,
	discount_factor=0.95, 
	pretrain_steps=100,
	batch_size=32,
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

```


### Structure explanation and implemented techniques

