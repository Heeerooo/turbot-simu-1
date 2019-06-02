import os
import pickle

from rl.memory import SequentialMemory


class SavableSequentialMemory(SequentialMemory):

    def __init__(self, limit, filename, **kwargs):
        super().__init__(limit, **kwargs)
        self.filename = filename

    def load(self):
        if os.path.isfile(self.filename):
            with open(self.filename, 'rb') as handle:
                loaded = pickle.load(handle)
                lenght = len(loaded['actions'])
                self.actions.length = lenght
                self.rewards.length = lenght
                self.terminals.length = lenght
                self.observations.length = lenght
                self.actions.data = loaded['actions']
                self.rewards.data = loaded['rewards']
                self.terminals.data = loaded['terminals']
                self.observations.data = loaded['observations']
                print("Memory loaded")

    def save(self):
        memory = {
            'actions': self.actions.data[:self.nb_entries],
            'rewards': self.rewards.data[:self.nb_entries],
            'terminals': self.terminals.data[:self.nb_entries],
            'observations': self.observations.data[:self.nb_entries]
        }
        with open(self.filename, 'wb') as handle:
            pickle.dump(memory, handle)

        print("Memory saved")

    def append(self, observation, action, reward, terminal, training=True):
        """Append an observation to the memory

        We override the parent method because we want to save data to memory even in test mode.

        # Argument
            observation (dict): Observation returned by environment
            action (int): Action taken to obtain this observation
            reward (float): Reward obtained by taking this action
            terminal (boolean): Is the state terminal
        """ 
        super(SavableSequentialMemory, self).append(observation, action, reward, terminal, training=training)
        
        # This needs to be understood as follows: in `observation`, take `action`, obtain `reward`
        # and weather the next state is `terminal` or not.
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)        

