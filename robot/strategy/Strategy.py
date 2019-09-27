from abc import ABC, abstractmethod


class Strategy(ABC):

    @abstractmethod
    def compute_steering(self):
        return None

    @abstractmethod
    def compute_speed(self):
        return None
