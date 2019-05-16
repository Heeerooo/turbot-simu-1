import os

import gym
import gym_simu

import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

ENV_NAME = 'simu-v0'
CHECKPOINT_WEIGHTS_FILE = 'dqn_simu-weights_checkpoint.h5f'
FINAL_WEIGHTS_FILE = 'dqn_simu-weights_one.h5f'

WINDOW_LENGTH = 4

class CustomRandomPolicy(EpsGreedyQPolicy):
    """
    This random policy is based on EpsGreedyPolicy,
    but each random action is followed by a random number of steps where:
       - either no action is taken
       - or Q action is taken

    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)

    """
    def __init__(self, eps=.1):
        super(CustomRandomPolicy, self).__init__(eps=eps)
        self.nb_without_action = 0  # Keeps the number of remaining steps without action to be taken
        self.MAX_STEPS_WITHOUT_ACTION = 8

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            if self.nb_without_action > 0:
                action = 0
            else:
                action = np.random.randint(0, nb_actions)
                self.nb_without_action = np.random.randint(0, self.MAX_STEPS_WITHOUT_ACTION)
        else:
            action = np.argmax(q_values)

        # Decrement number of steps without action
        self.nb_without_action -= 1
        self.nb_without_action = max(self.nb_without_action, 0)

        return action

# for i in range(10):
env = gym.make(ENV_NAME)
# Get the environment and extract the number of actions.
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=250000, window_length=WINDOW_LENGTH)
# policy = BoltzmannQPolicy()
# policy = EpsGreedyQPolicy(eps=0.1)
# policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1000000)

policy = CustomRandomPolicy(eps=1.0)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if os.path.isfile(CHECKPOINT_WEIGHTS_FILE):
    dqn.load_weights(CHECKPOINT_WEIGHTS_FILE)

# dqn.test(env, nb_episodes=5, visualize=False)

dqn.fit(env, nb_steps=40000, visualize=False, verbose=2, nb_max_episode_steps=200)

# After training is done, we save the final weights.
dqn.save_weights(FINAL_WEIGHTS_FILE, overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
env.close()
