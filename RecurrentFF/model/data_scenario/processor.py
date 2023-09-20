from abc import ABCMeta, abstractmethod
from enum import Enum

import torch
from RecurrentFF.settings import Settings

from RecurrentFF.util import TrainLabelData


class DataScenario(Enum):
    StaticSingleClass = 1


class DataScenarioProcessor(metaclass=ABCMeta):
    @abstractmethod
    def brute_force_predict(self):
        pass

    @abstractmethod
    def train_class_predictor_from_latents(
            self,
            latents: torch.Tensor,
            labels: torch.Tensor,
            total_batch_count: int):
        pass

    @abstractmethod
    def replace_negative_data_inplace(
            self,
            input_batch: torch.Tensor,
            input_labels: TrainLabelData,
            total_batch_count: int):
        pass

    @abstractmethod
    def get_preinit_upper_clamped_tensor(
            self, upper_clamed_tensor_shape: tuple):
        pass
