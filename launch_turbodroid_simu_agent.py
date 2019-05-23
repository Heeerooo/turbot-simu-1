import os

import gym
import gym_simu

import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy

from custom_policy.TurbodroidRandomPolicy import TurbodroidPolicyRepeat

ENV_NAME = 'simu-v0'
CHECKPOINT_WEIGHTS_FILE = 'dqn_simu-weights_checkpoint.h5f'
PARAMS_FILE = 'training_parameters.npy'

# Constants for Annealed random policy
START_EPSILON = 1.0
END_EPSILON = 0.1
EPSILON_TEST = 0.05
NUM_STEPS_ANNEALED = 2000000    # Nb steps to bring epsilon from start epsilon to end epsilon
NUM_STEPS_BEFORE_RESET = 40000    # Reset every N steps because of memory leak
WINDOW_LENGTH = 4

# for i in range(10):
env = gym.make(ENV_NAME)
# Get the environment and extract the number of actions.
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
# print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=250000, window_length=WINDOW_LENGTH)
# policy = BoltzmannQPolicy()
# policy = EpsGreedyQPolicy(eps=0.1)

# Load parameters from file if exists
if os.path.isfile(PARAMS_FILE):
    eps = np.load(PARAMS_FILE)
else:
    eps = START_EPSILON

# policy = CustomRandomPolicy(eps=1.0)
delta_eps = (START_EPSILON - END_EPSILON) * NUM_STEPS_BEFORE_RESET / NUM_STEPS_ANNEALED
next_eps = max(eps - delta_eps, END_EPSILON)
print("Epsilon: ", eps, "Next epsilon: ", next_eps)
# policy = LinearAnnealedPolicy(TurbodroidRandomPolicy(), attr='eps', value_max=eps, value_min=next_eps, value_test=EPSILON_TEST, nb_steps=NUM_STEPS_BEFORE_RESET)
policy = LinearAnnealedPolicy(TurbodroidPolicyRepeat(), attr='eps', value_max=eps, value_min=next_eps, value_test=EPSILON_TEST, nb_steps=NUM_STEPS_BEFORE_RESET)


dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               train_interval=1, target_model_update=1000, gamma=.93, policy=policy)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if os.path.isfile(CHECKPOINT_WEIGHTS_FILE):
    dqn.load_weights(CHECKPOINT_WEIGHTS_FILE)
    print("Checkpoint file loaded")

# dqn.test(env, nb_episodes=5, visualize=False)
tbCallBack = TensorBoard(log_dir='./logs/model10_wall_penalty0.4_without_ressort_steering')
dqn.fit(env, nb_steps=NUM_STEPS_BEFORE_RESET, visualize=False, verbose=1, nb_max_episode_steps=200, callbacks=[tbCallBack])

# After training is done, we save the final weights.
dqn.save_weights(CHECKPOINT_WEIGHTS_FILE, overwrite=True)

# Save next epsilon to parameters file
np.save(PARAMS_FILE, next_eps)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
env.close()
