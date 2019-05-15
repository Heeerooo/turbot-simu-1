import os

import gym
import numpy as np
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

ENV_NAME = 'simu-v0'


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
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

if os.path.isfile('dqn_simu-v0_weights_one.h5f'):
    dqn.load_weights('dqn_simu-v0_weights_one.h5f')

dqn.test(env, nb_episodes=5, visualize=False)

dqn.fit(env, nb_steps=40000, visualize=False, verbose=2, nb_max_episode_steps=200)

# After training is done, we save the final weights.
dqn.save_weights('dqn_simu-v0_weights_one.h5f', overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
env.close()
