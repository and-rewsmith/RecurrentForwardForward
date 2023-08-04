import toml
import torch

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
        def __init__(self, model_dict):
            self.hidden_sizes = model_dict['hidden_sizes']
            self.epochs = model_dict['epochs']
            self.loss_threshold = model_dict['loss_threshold']
            self.damping_factor = model_dict['damping_factor']
            self.epsilon = model_dict['epsilon']
            self.learning_rate = model_dict['learning_rate']
            self.skip_profiling = model_dict['skip_profiling']

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
        config_file = args.config_file if args.config_file != None else CONFIG_FILE
        return Settings.from_config_file(config_file)
