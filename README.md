# Skopos - What do you want me to do?

<img src="img/logo.png" width="50%">

Skopos is a flexible Deep Reinforcement Learning Library which gives the possibility to the end user to implement and test faster the different Deep Reinforcement Learning techniques. The structure is based on hierarchical blocks; you can plug different techniques and experiment different combinations of “blocks”. For example using an AsynchronousAgent and AdvantageActorCritic as learning technique you generate A3C method.

### Library main assumption

The given environment must have the same characteristics as an OpenAI Gym environment. The methods used in the library are:

```python
env.reset()
```

To restart the episode.

```python
env.action_space.n
``` 

To get the number of possible action.

```python
env.step(action)
```

To pass from a state to another. 

```python 
env.observation_space.shape[i]
``` 

To get the input state dimension.

### Installation and requirements

The requirements are:

- python=2.7
- tensorflow>=1.4.0
- numpy>=1.13.1
- matplotlib>=2.0.2
- scipy>=1.0.0

Currently there is a problem with matplotlib. Before installing skopos, install matplotlib from conda.

	$ conda install matplotlib

Then you can clone the repository or install it from pip.

	$ pip install skopos 

If you want to use an OpenAI Gym environment, install gym from pip using:

	$ pip install gym

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


# Structure explanation and implemented techniques

## Documentation

The documentation tries to give you a brief intuition regarding each method with problems, advantages and how to use it from the code point of view.

### Agent

The agent is the Game Player. It “plays” many games in order to learn the best strategy to get more rewards.

#### SimpleAgent

One single agent (network) which explores and takes action in the environment.

- Advantage: off-policy (the current network parameters are different to those used to generate the sample); it learns about the greedy strategy, while following a behaviour distribution that ensures adequate exploration of the state space.

#### AsynchronousAgent

It asynchronously executes multiple agents in parallel, on multiple instances of the environment. 

- Advantage: no need of GPU. This idea enables a much larger spectrum of fundamental on-policy RL algorithms, such as Sarsa, n-step methods, and actor- critic methods, as well as off-policy RL algorithms such as Q-learning, to be applied robustly and effectively using deep neural networks. 

### Policy

A Policy is the agent’s strategy to choose an action at each state. We focused our attention just on Deep Reinforcement Learning techniques, which means NetworkBasedPolicy. For testing proposal we also implemented the RandomPolicy method.

### Learner

We introduced the concept of learner to identify the method used to update the NetworkBasedPolicy at each train step. There are two different types of learners:

#### ValueIterationLearner

The update is done using V(S) and Q(S, a). [intuition: if we know Q for every state and action, on every state we will choose the best action.]

##### DQN

A deep Q network (DQN) is a multi-layered neural network that for a given state s outputs a vector of action values Q(s, · ; θ), where θ are the parameters of the network. 

- Problem: it suffers from substantial overestimations of the action values under certain conditions. To avoid this problem use double_agent = True which split the generation of actions (max(Q(S’, a)) to the estimation of the TargetQ values.

- Output: action value Q(S, a) for each action starting from state S.

- Loss Function: 

##### DoubleAgent

Two different networks in the same environment. The main network θt, computes the action and explore the environment while the target θt’ one generates the TargetQ values for the training process. 

- Advantage: value estimates are more accurate, improving the performances in terms of scores.

##### Dueling

The dueling architecture consists of two streams that represent the value V(S) and advantage A(S, a) functions, while sharing a common learning model. The two streams are combined via a special aggregating layer to produce an estimate of the state-action value function Q(S, a). 

- Advantage: the main benefit of this factoring is to generalise learning across actions without imposing any change to the underlying reinforcement learning algorithm. Intuitively, the dueling architecture can learn which states are (or are not) valuable, without having to learn the effect of each action for each state. This is particularly useful in states where its actions do not affect the environment in any relevant way. 

- Output: action value Q(S, a) for each action starting from state S, generated by summing the two streams: V(S) and A(S, a).

- Loss Function: 

##### NStepQLearning

The target q is generated using the discounted rewards in a certain episode.

- Advantage: it keeps track of the rewards generated during the episode.

- Output: action value Q(S, a) for each action starting from state S.

- Loss Function: 

#### PolicyGradientLearner

The update is done using the policy which is represented by the neural network a = pi(a/S, u) where u are the weights. It defines the objective function as the total discounted reward. [intuition: we know the outcome of our actions, promote good actions and punish bad ones.]

#### VanillaPolicyGradient

The policy is learnt iteratively updating the network weights after the finish of each episode considering instead of the TargetQ value, the discounted rewards of the episode.

- Problem: the main problem regards the variance, which means that we take a lot of steps in poor directions, even though on average the step will be in the correct direction. In order to avoid this kind of problem ActorCritic is a more suitable method.

- Output: probability to take each action after applying to action value function Q(S, a) a Soft-max layer.

#### AdvantageActorCritic

It approximates policy gradient, using the Advantage function A(S, a) instead of the discounted rewards Rt, to decrease the variance of the gradients.

- Advantage: decrease the variance.

- Problem: approximating the policy gradient introduces bias, but if we choose value function approximation carefully we can avoid introducing any bias.

- Output: policy (actions probabilities) and value of the state.

- Loss Function: the error function is composed by policy loss, value loss and entropy (adding entropy to the loss function was found to improve exploration by limiting the premature convergence to suboptimal policy). A represents the Advantage function (Q(S, a) - V(S)) which corresponds to (Rt - Bt(St)) where Bt is the estimation of V(S) and it is used to decrease the bias. 

### Exploration Strategy

The exploration strategy is the way in which the agent tries to avoid local optimal, keeping a certain rate of randomness in the action decision making.

#### Greedy

It takes always the best action.

#### E-Greedy

The agent follows the policy but with a certain probability e it takes a random action.

#### Decremental E-Greedy

The basic functioning is the same of the E-Greedy exploration strategy but for the fact that the probability e is decreased over time. This allows e to be set to a greater initial value that is then annealed over time. In this way the exploration can be pushed in the early stages before turning the strategy towards more exploitative choices.

#### Boltzman

The action is chosen among the possible actions with a probability that is the softmax of the expected rewards. Given the expected reward for the action j, rj, the action is taken with a probability of exp(rj/e) / sum(exp(rk/e)) where k varies over all the possible actions. The parameter e is annealed over time, once more, to manage the exploration vs exploitation trade off in favour of a stronger initial explorative attitude that is given up with the progress of the learning.

### Memory

#### Experience Replay 

At each step we store the agent Experience = (St, at, rt, St+1, Done). The training process takes a batch of this experience to learn the optimal policy.

- Advantage: it allows data efficiency, and increase generalisation. Learning directly from consecutive samples is inefficient; randomising the samples breaks these correlations and therefore reduces the variance of the updates. 
Problem: aggregating over memory in this way reduces non-stationarity and decor-relates updates, but at the same time limits the methods to off-policy reinforcement learning algorithms, moreover it uses more memory and computation per real interaction; and it requires off-policy learning algorithms that can update from data generated by an older policy. The solution is to use AsynchronousAgent which executes multiple agents in parallel, on multiple instances of the environment. 

#### Prioritised Experience Replay

The key idea is to increase the replay probability of experience that are likely to improve as much as possible the learning.

- Advantage: it leads to both faster learning and to better final policy quality.

Different prioritisation strategies are available to define the priority with which a certain episode should be selected for replay.

- Average expectation error over the episode: is the average value of the distance between the computed expected reward for the step and the actual received reward for the given episode.
- Maximum expectation error over the episode: the same as above but instead of considering the average we just choose the maximum estimation error.
- Closeness of the expected rewards: prioritises the replay of those episodes close to the decision boundaries and for which, thus, the policy expresses close expected rewards for the different actions. It may be interesting to learn how to distinguish the “ambiguous cases” first. The closeness is measured ordering the array of the expected rewards in the sequence  r0,…,rn of length n + 1. The closeness is given as the sum of ri/ri+1 divided by i+1 starting from the first non zero value. At the end the number is divided by the sum of the i+1 of the valid steps. Reward sequences in which there’s not much difference between the values will produce priorities that tend to 1, and viceversa well distinguished cases will produce values that are close to zero.

### Visualisation and Training Metrics

In the end of the training the user can see the Agent performances using the methods in the skopos.util.visualize.py file. The used metrics/plot are:

- Average reward per episode: in order to show the learning during the training.

- Number of actions per episode: in order to show that going further in the learning, the number of actions taken in an episode should decrease/increase based on the game obkective.

- Average estimated Q-Values per episode: it provides an estimate of how much discounted reward the agent can obtain by following its policy from any given state.  

Or alternatevely using Tensorboard (tensorboard_visualization=True) with the following command:

	$ tensorboard --logdir=tensorboard/train/








