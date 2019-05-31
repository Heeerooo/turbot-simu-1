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

