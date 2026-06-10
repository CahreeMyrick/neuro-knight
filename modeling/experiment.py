from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

