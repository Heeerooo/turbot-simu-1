import gym
import gym_simu

# env = gym.make('simu-v0')

# for _ in range(30):
#     env.render()
#     ob, reward, episode_over, _ = env.step(env.action_space.sample()) # take a random action
#     print(ob)
    # env.step(None) # take a random action

# print("resetting")
# env.reset()

# for _ in range(30):
#     env.render()
#     env.step(env.action_space.sample()) # take a random action
#     # env.step(None) # take a random action


# env.close()

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'simu-v0'


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
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
memory = SequentialMemory(limit=50000, window_length=1)
# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

for i in range(5):
    dqn.fit(env, nb_steps=10000, visualize=False, verbose=1, nb_max_episode_steps=200)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    dqn.test(env, nb_episodes=5, visualize=False)
