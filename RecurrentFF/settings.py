import toml
import torch

from pydantic import BaseModel

from RecurrentFF.benchmarks.arguments import get_arguments

CONFIG_FILE = "./config.toml"


# NOTE: No mutable state allowed. Everything should be static if using this, so
# singleton ok.
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Settings(metaclass=Singleton):
    class Model:
        class FfRmsprop:
            def __init__(self, ff_rmsprop_dict):
                self.momentum = ff_rmsprop_dict['momentum']
                self.learning_rate = ff_rmsprop_dict['learning_rate']

        class ClassifierRmsprop:
            def __init__(self, classifier_rmsprop_dict):
                self.momentum = classifier_rmsprop_dict['momentum']
                self.learning_rate = classifier_rmsprop_dict['learning_rate']

        class FfAdam:
            def __init__(self, ff_adam_dict):
                self.learning_rate = ff_adam_dict['learning_rate']
        
        class ClassifierAdam:
            def __init__(self, classifier_adam_dict):
                self.learning_rate = classifier_adam_dict['learning_rate']

        def __init__(self, model_dict):
            self.hidden_sizes = model_dict['hidden_sizes']
            self.epochs = model_dict['epochs']
            self.loss_threshold = model_dict['loss_threshold']
            self.damping_factor = model_dict['damping_factor']
            self.epsilon = model_dict['epsilon']
            self.skip_profiling = model_dict['skip_profiling']
            self.should_log_metrics = model_dict["should_log_metrics"]
            self.should_replace_neg_data = model_dict["should_replace_neg_data"]

            self.ff_optimizer = model_dict['ff_optimizer']
            self.classifier_optimizer = model_dict['classifier_optimizer']
            

            if self.ff_optimizer == "rmsprop":
                self.ff_rmsprop = self.FfRmsprop(model_dict['ff_rmsprop'])
            elif self.ff_optimizer == "adam":
                self.ff_adam = self.FfAdam(model_dict['ff_adam'])

            if self.classifier_optimizer == "rmsprop":
                self.classifier_rmsprop = self.ClassifierRmsprop(model_dict['classifier_rmsprop'])
            elif self.classifier_optimizer == "adam":
                self.classifier_adam = self.ClassifierAdam(model_dict['classifier_adam'])

    class Device:
        def __init__(self, device_dict):
            self.device = torch.device(device_dict['device'])

    def __init__(self, config: dict[str, any]):
        self.model = self.Model(config['model'])
        self.device = self.Device(config['device'])

    def from_config_file(path: str):
        config = toml.load(path)
        return Settings(config)

    def new():
        args = get_arguments()
        config_file = args.config_file if args.config_file is not None else CONFIG_FILE
        return Settings.from_config_file(config_file)
