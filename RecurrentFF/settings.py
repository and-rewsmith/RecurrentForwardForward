import toml
import torch

from pydantic import BaseModel

from RecurrentFF.benchmarks.arguments import get_arguments

CONFIG_FILE = "./config.toml"


# NOTE: No mutable state allowed. Everything should be static if using this, so
# singleton ok.
class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]


class FfRmsprop(BaseModel):
    momentum: float
    learning_rate: float


class ClassifierRmsprop(BaseModel):
    momentum: float
    learning_rate: float


class FfAdam(BaseModel):
    learning_rate: float


class ClassifierAdam(BaseModel):
    learning_rate: float


class Model(BaseModel):
    hidden_sizes: list
    epochs: int
    loss_threshold: float
    damping_factor: float
    epsilon: float
    skip_profiling: bool
    should_log_metrics: bool
    should_replace_neg_data: bool
    ff_optimizer: str
    classifier_optimizer: str
    ff_rmsprop: FfRmsprop = None
    ff_adam: FfAdam = None
    classifier_rmsprop: ClassifierRmsprop = None
    classifier_adam: ClassifierAdam = None


class Device(BaseModel):
    device: str  # You may wish to modify this to suit your needs


class Settings(BaseModel, Singleton):
    model: Model
    device: Device

    @classmethod
    def from_config_file(cls, path: str):
        config = toml.load(path)
        model = config['model']

        if model['ff_optimizer'] == "rmsprop":
            model['ff_rmsprop'] = FfRmsprop(**model['ff_rmsprop'])
            model['classifier_rmsprop'] = ClassifierRmsprop(
                **model['classifier_rmsprop'])
        elif model['ff_optimizer'] == "adam":
            model['ff_adam'] = FfAdam(**model['ff_adam'])
            model['classifier_adam'] = ClassifierAdam(
                **model['classifier_adam'])

        return cls(model=Model(**model), device=Device(**config['device']))

    @classmethod
    def new(cls):
        args = get_arguments()
        config_file = args.config_file if args.config_file is not None else CONFIG_FILE
        return cls.from_config_file(config_file)
