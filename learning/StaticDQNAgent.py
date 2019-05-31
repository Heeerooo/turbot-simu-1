import numpy as np
from rl.agents import DQNAgent


class StaticDQNAgent(DQNAgent):

    def backward_without_memory(self, batch_indexes):

        state0_batch, targets, dummy_targets, masks = self.do_backward(batch_indexes)

        ins = [state0_batch] if type(self.model.input) is not list else state0_batch
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
        metrics += self.policy.metrics
        if self.processor is not None:
            metrics += self.processor.metrics

        return metrics

    def evaluate_on_batch(self, batch_indexes):

        state0_batch, targets, _, masks = self.do_backward(batch_indexes)

        ins = [state0_batch] if type(self.model.input) is not list else state0_batch
        pred = self.trainable_model.predict_on_batch(ins + [targets, masks])

        # Loss is first output of trainable_model
        batch_losses = pred[0]

        return batch_losses

    def do_backward(self, batch_indexes):
        # Train the network on a single stochastic batch.
        experiences = self.memory.sample(self.batch_size, batch_idxs=batch_indexes)
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

        return state0_batch, targets, dummy_targets, masks

    def process_on_dataset_by_batch(self, action, results, dataset_indexes):
        batch_indexes = []
        for i in dataset_indexes:
            if len(batch_indexes) < self.batch_size:
                # Fill in batch
                batch_indexes.append(i)
            else:
                # Do backward on batch
                metrics = action(batch_indexes)
                results.append(metrics)
                batch_indexes = []
