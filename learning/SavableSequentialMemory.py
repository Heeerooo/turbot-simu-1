import os

import numpy as np
from rl.memory import SequentialMemory


class SavableSequentialMemory(SequentialMemory):

    def __init__(self, limit, filename, **kwargs):
        super().__init__(limit, **kwargs)
        self.filename = filename

    def load(self):
        if os.path.isfile('dqn_simu_memory.npz'):
            loaded = np.load("dqn_simu_memory.npz")
            memory = loaded["memory"]
            lenght = len(memory[0])
            self.actions.length = lenght
            self.rewards.length = lenght
            self.terminals.length = lenght
            self.observations.length = lenght
            self.actions.data = memory[0].tolist()
            self.rewards.data = memory[1].tolist()
            self.terminals.data = memory[2].tolist()
            self.observations.data = memory[3].tolist()
            print("Memory loaded")

    def save(self):
        memory = np.array([self.actions.data[:self.nb_entries], self.rewards.data[:self.nb_entries],
                  self.terminals.data[:self.nb_entries], self.observations.data[:self.nb_entries]])
        np.savez_compressed("dqn_simu_memory.npz", memory=memory)
        print("Memory saved")

