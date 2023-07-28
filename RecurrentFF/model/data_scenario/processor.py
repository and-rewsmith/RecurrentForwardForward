from abc import ABCMeta, abstractmethod


class DataScenarioProcessor(metaclass=ABCMeta):
    @abstractmethod
    def brute_force_predict(self):
        pass
