import os

import gym
import gym_simu
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy
import random

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


class CustomDQNAgent(DQNAgent):

    def backward_without_memory(self, reward, terminal):

        metrics = [np.nan for _ in self.metrics_names]

        # TODO I removed a if here, check if needed

        # Train the network on a single stochastic batch.
        experiences = self.memory.sample(self.batch_size)
        assert len(experiences) == self.batch_size

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        # Compute Q values for mini-batch update.
        if self.enable_double_dqn:
            # According to the paper "Deep Reinforcement Learning with Double Q-learning"
            # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
            # while the target network is used to estimate the Q value.
            q_values = self.model.predict_on_batch(state1_batch)
            assert q_values.shape == (self.batch_size, self.nb_actions)
            actions = np.argmax(q_values, axis=1)
            assert actions.shape == (self.batch_size,)

            # Now, estimate Q values using the target network but select the values with the
            # highest Q value wrt to the online model (as computed above).
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            assert target_q_values.shape == (self.batch_size, self.nb_actions)
            q_batch = target_q_values[range(self.batch_size), actions]
        else:
            # Compute the q_values given state1, and extract the maximum for each sample in the batch.
            # We perform this prediction on the target_model instead of the model for reasons
            # outlined in Mnih (2015). In short: it makes the algorithm more stable.
            target_q_values = self.target_model.predict_on_batch(state1_batch)
            assert target_q_values.shape == (self.batch_size, self.nb_actions)
            q_batch = np.max(target_q_values, axis=1).flatten()
        assert q_batch.shape == (self.batch_size,)

        targets = np.zeros((self.batch_size, self.nb_actions))
        dummy_targets = np.zeros((self.batch_size,))
        masks = np.zeros((self.batch_size, self.nb_actions))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * q_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch
        assert discounted_reward_batch.shape == reward_batch.shape
        Rs = reward_batch + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[action] = R  # update action with estimated accumulated reward
            dummy_targets[idx] = R
            mask[action] = 1.  # enable loss for this specific action
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        ins = [state0_batch] if type(self.model.input) is not list else state0_batch
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
        metrics += self.policy.metrics
        if self.processor is not None:
            metrics += self.processor.metrics

        # TODO if we want to update hard
        # if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
        #     self.update_target_model_hard()

        return metrics

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

dqn = CustomDQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               train_interval=1, target_model_update=0.1, gamma=.93, policy=policy)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if os.path.isfile(CHECKPOINT_WEIGHTS_FILE):
    dqn.load_weights(CHECKPOINT_WEIGHTS_FILE)
    print("Checkpoint file loaded")

memory.load()

tbCallBack = TensorBoard(log_dir='./logs/test_async_training4')
tbCallBack.set_model(model)
tbCallBack.on_train_begin()

# Train / test split
nb_entries = memory.nb_entries
indexes = list(range(nb_entries))
nb_entries_train = int(nb_entries * 2 / 3)
indexes_train = indexes[:nb_entries_train]
indexes_val = indexes[nb_entries_train:]

print("Size of dataset: ", nb_entries)

for j in range(1000):
    print("Epoch ", j)
    metrics_list = []
    tbCallBack.on_epoch_begin(j)    

    # Shuffle train indexes
    random.shuffle(indexes_train)

    print("Training on ", len(indexes_train), " train indexes")

    for i in indexes_train:
        dqn.recent_action = memory.actions[i]
        dqn.recent_observation = memory.observations[i]

        metrics = dqn.backward_without_memory(memory.rewards[i], memory.terminals[i])
        metrics_list.append(metrics)

    # Update target_model
    dqn.update_target_model_hard()
        
    # Compute mean of metrics for epoch
    epoch_metrics = np.array(metrics_list).mean(axis=0)
    
    # Log metrics for tensorboard
    epoch_logs = dict(zip(dqn.metrics_names, epoch_metrics))
    tbCallBack.on_epoch_end(j, logs=epoch_logs)

tbCallBack.on_train_end(None)

env.close()
