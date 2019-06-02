import os
import random

import gym
import gym_simu
import numpy as np
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
# from rl.policy import LinearAnnealedPolicy

from learning.SavableSequentialMemory import SavableSequentialMemory
from learning.StaticDQNAgent import StaticDQNAgent
from learning.TurbodroidModel import TurbodroidModel
from learning.TurbodroidRandomPolicy import TurbodroidPolicyRepeat

ENV_NAME = 'simu-v0'
CHECKPOINT_WEIGHTS_FILE = 'dqn_simu-weights_checkpoint.h5f'
PARAMS_FILE = 'training_parameters.npy'
MEMORY_FILE= 'dqn_simu_memory.pickle'

NB_EPOCHS = 50

# Constants for Annealed random policy
START_EPSILON = 1.0
END_EPSILON = 0.1
EPSILON_DECAY = 0.05  # Decay of epsilon at each step
NUM_EPISODES_PER_LOOP = 50  # Nb of episodes to generate in each dataset
WINDOW_LENGTH = 4


#################################
# Initialisation
#################################

# Create the simulation environment
env = gym.make(ENV_NAME)
# Get the environment and extract the number of actions.
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build our model.
model = TurbodroidModel(WINDOW_LENGTH, env.observation_space.shape, nb_actions).get()

# Create the memory object
memory = SavableSequentialMemory(limit=250000, filename=MEMORY_FILE, window_length=WINDOW_LENGTH)

################################
# Create the policy
################################

# Load parameters from file if exists
if os.path.isfile(PARAMS_FILE):
    eps = np.load(PARAMS_FILE)
else:
    eps = START_EPSILON

next_eps = max(eps - EPSILON_DECAY, END_EPSILON)
print("Epsilon: ", eps, "Next epsilon: ", next_eps)

policy = TurbodroidPolicyRepeat(eps=eps)

################################
# Create the DQN agent
################################

# We are going to use test mode to generate data, as we do the training asynchronously
dqn = StaticDQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
                     train_interval=1, target_model_update=0.1, gamma=.93, test_policy=policy)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

# Load model previously saved if exists
if os.path.isfile(CHECKPOINT_WEIGHTS_FILE):
    dqn.load_weights(CHECKPOINT_WEIGHTS_FILE)
    print("Checkpoint file loaded")

################################
# Create the training dataset
################################

# Load previously created data
# memory.load() # TODO: fix this. It does not work. If we load, then append fails afterwards.

# Generate more data by running the simulation
dqn.test(env, nb_episodes=NUM_EPISODES_PER_LOOP, visualize=False, verbose=1, nb_max_episode_steps=200)

print("Memory length: ", memory.actions.length )

################################
# Train on the generated dataset
################################

# Initialize Tensorboard
tbCallBack = TensorBoard(log_dir='./logs/test_async_training6')
tbCallBack.set_model(model)
tbCallBack.on_train_begin()

# Train / test split
nb_entries = memory.nb_entries
indexes = list(range(memory.window_length, nb_entries - 1))
nb_entries_train = int(nb_entries * 2 / 3)
indexes_train = indexes[:nb_entries_train].copy()
indexes_val = indexes[nb_entries_train:].copy()

print("Size of dataset: ", nb_entries)

for j in range(NB_EPOCHS):
    print("Epoch ", j)

    # Init epoch
    metrics_list = []
    val_loss_list = []
    tbCallBack.on_epoch_begin(j)
    random.shuffle(indexes_train)

    # Do backward on dataset
    dqn.process_on_dataset_by_batch(dqn.backward_without_memory, metrics_list, indexes_train)

    # Update target_model
    dqn.update_target_model_hard()

    # Compute mean of train metrics for epoch
    epoch_metrics = np.array(metrics_list).mean(axis=0)

    # Evaluate on validation dataset
    dqn.process_on_dataset_by_batch(dqn.evaluate_on_batch, val_loss_list, indexes_val)

    # Log metrics for tensorboard
    epoch_val_loss_mean = np.array(val_loss_list).mean()
    epoch_logs = dict(zip(dqn.metrics_names, epoch_metrics))
    epoch_logs['val_loss'] = epoch_val_loss_mean
    print(epoch_logs)
    tbCallBack.on_epoch_end(j, logs=epoch_logs)

tbCallBack.on_train_end(None)

# TODO: keep the model that got the lowest validation loss
dqn.save_weights(CHECKPOINT_WEIGHTS_FILE, overwrite=True)
env.close()

# Save next epsilon to parameters file
np.save(PARAMS_FILE, next_eps)
