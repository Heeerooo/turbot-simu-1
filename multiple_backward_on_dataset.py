import os

import gym
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy

from learning.SavableSequentialMemory import SavableSequentialMemory
from learning.TurbodroidRandomPolicy import TurbodroidPolicyRepeat

ENV_NAME = 'simu-v0'
CHECKPOINT_WEIGHTS_FILE = 'dqn_simu-weights_checkpoint.h5f'
PARAMS_FILE = 'training_parameters.npy'

# Constants for Annealed random policy
START_EPSILON = 1.0
END_EPSILON = 0.1
EPSILON_TEST = 0.05
NUM_STEPS_ANNEALED = 2000000  # Nb steps to bring epsilon from start epsilon to end epsilon
NUM_STEPS_BEFORE_RESET = 100  # Reset every N steps because of memory leak
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


memory = SavableSequentialMemory(limit=250000, filename="dqn_simu_memory.npz", window_length=WINDOW_LENGTH)


# Load parameters from file if exists
if os.path.isfile(PARAMS_FILE):
    eps = np.load(PARAMS_FILE)
else:
    eps = START_EPSILON

delta_eps = (START_EPSILON - END_EPSILON) * NUM_STEPS_BEFORE_RESET / NUM_STEPS_ANNEALED
next_eps = max(eps - delta_eps, END_EPSILON)
print("Epsilon: ", eps, "Next epsilon: ", next_eps)
policy = LinearAnnealedPolicy(TurbodroidPolicyRepeat(), attr='eps', value_max=eps, value_min=next_eps,
                              value_test=EPSILON_TEST, nb_steps=NUM_STEPS_BEFORE_RESET)

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               train_interval=1, target_model_update=1000, gamma=.93, policy=policy)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if os.path.isfile(CHECKPOINT_WEIGHTS_FILE):
    dqn.load_weights(CHECKPOINT_WEIGHTS_FILE)
    print("Checkpoint file loaded")

memory.load()

tbCallBack = TensorBoard(log_dir='./logs/test_async_training')

for i in range(0, memory.nb_entries):
    dqn.recent_action = memory.actions[i]
    dqn.recent_observation = memory.observations[i]
    metrics = dqn.backward(memory.rewards[i], memory.terminals[i])

    step_logs = {
        'metrics': metrics,
    }
    tbCallBack.on_batch_end(i, step_logs)

env.close()
