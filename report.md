# Report
---
This project is implemented based on [deep-reinforcement-learning/p3_collab-compet](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet).


## Algorithm

The models are training `ddpg` function in the `Continuous_Control` notebook.
The DDPG algorithm has two parts as shown below:
![DDPG1](https://github.com/ainilaha/Continuous-Control/blob/master/images/ddpg1.jpg?raw=true)
With soft update method as below:
![soft update](https://github.com/ainilaha/Continuous-Control/blob/master/images/soft_update.jpg?raw=true)

- `first score and its window`
- `initialize the agents`
- `loop over the agents`
- `training the models episodically with limited num of episodes`
- `the training process will stop when score is reached +30`
- `reset the agent in every episode`
- `save the scores in every episode`
- `The agents have its Actors but share same critic`

more detail can shown the algorithm from the [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf):
![soft update](https://github.com/ainilaha/Continuous-Control/blob/master/images/ddpg_alg.jpg?raw=true)

The DDPG agent is implemented in `ddpg_agent.py`

### DDPG Hyper Parameters
- `n_episodes=13500`: maximum number of training episodes,
- `num_agents`: number of agents

### DDPG Agent Hyper Parameters

- `BUFFER_SIZE = int(1e5)`: replay buffer size
- `BATCH_SIZE = 128`: mini batch size
- `GAMMA = 0.99`: discount factor
- `TAU = 1e-3`: for soft update of target parameters
- `LR_ACTOR = 1e-4`: learning rate for optimizer
- `LR_CRITIC = 1e-4`: learning rate for optimizer
- `WEIGHT_DECAY = 0.0`: L2 weight decay
- `N_LEARN_UPDATES = 10`: number of learning updates
- `N_TIME_STEPS = 20`: every n time step do update


### Neural Networks

Actor and Critic network models were defined in `model.py`

The Actor network and critic network set have three layers.
The first layer is input layer with size as input size.
The second and last layer has 512 and 256 neruals.

## Plot of rewards
![Tennis](https://github.com/ainilaha/collab-compet/blob/master/images/tennis.png?raw=true?raw=true)

```

```

## Ideas for Future Work

Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) methods could leverage in the future work on this project.

In addition, the neural networks in current model are rather relatively shallow; therefore, we can make it a little more deep.

Lastly, due to the computational resource, the crawler was only trained with score +0.5. The performance of the controller of the agent probably can been improved if given more time and resources to train.  
