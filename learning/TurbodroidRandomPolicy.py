import numpy as np
from rl.policy import EpsGreedyQPolicy


class TurbodroidRandomPolicy(EpsGreedyQPolicy):
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
        super(TurbodroidRandomPolicy, self).__init__(eps=eps)
        self.nb_without_action = 0  # Keeps the number of remaining steps without action to be taken
        self.MAX_STEPS_WITHOUT_ACTION = 5

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

class TurbodroidPolicyRepeat(EpsGreedyQPolicy):
    """
    This random policy is based on EpsGreedyPolicy,
    but each action is followed by a 3 steps where the same action is taken

    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)

    """
    def __init__(self, eps=.1):
        super(TurbodroidPolicyRepeat, self).__init__(eps=eps)
        self.NB_REPEAT_STEPS = 4
        self.counter = self.NB_REPEAT_STEPS - 1
        self.last_action = 0

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1

        self.counter += 1
        if self.counter == self.NB_REPEAT_STEPS:
            # Reset counter
            self.counter = 0
            # Select the new action

            nb_actions = q_values.shape[0]

            if np.random.uniform() < self.eps:
                # Random action
                action = np.random.randint(0, nb_actions)
                self.last_action = action
            else:
                # max Q action
                action = np.argmax(q_values)

            return action

        else:
            # Repeat last action
            return self.last_action    

class TurbodroidPolicyRepeatVariant(EpsGreedyQPolicy):
    """
    This random policy is based on EpsGreedyPolicy,
    but each random action is memorized for the next 3 steps, to repeat it

    Eps Greedy policy either:
    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)

    """
    def __init__(self, eps=.1):
        super(TurbodroidPolicyRepeatVariant, self).__init__(eps=eps)
        self.NB_REPEAT_STEPS = 4
        self.counter = self.NB_REPEAT_STEPS - 1
        self.last_action = 0

    def select_action(self, q_values):
        """Return the selected action
        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action
        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        self.counter += 1

        if np.random.uniform() < self.eps:

            if self.counter >= self.NB_REPEAT_STEPS:
                # Reset counter
                self.counter = 0
                # Select the new random action
                action = np.random.randint(0, nb_actions)
                # Memorize it for next steps
                self.last_action = action
            else:
                # We repeat last random action
                action = self.last_action

        else:
            # max Q action
            action = np.argmax(q_values)

        return action
