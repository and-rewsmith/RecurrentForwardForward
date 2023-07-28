import toml


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Settings(metaclass=Singleton):
    class Data:
        def __init__(self, data_dict):
            self.train_batch_size = data_dict['train_batch_size']
            self.test_batch_size = data_dict['test_batch_size']

    class Model:
        def __init__(self, model_dict):
            self.epochs = model_dict['epochs']
            self.loss_threshold = model_dict['loss_threshold']
            self.damping_factor = model_dict['damping_factor']
            self.epsilon = model_dict['epsilon']
            self.learning_rate = model_dict['learning_rate']
            self.skip_profiling = model_dict['skip_profiling']
            self.default_focus_iteration_neg_offset = model_dict['default_focus_iteration_neg_offset']
            self.default_focus_iteration_pos_offset = model_dict['default_focus_iteration_pos_offset']

    class Device:
        def __init__(self, device_dict):
            self.device = device_dict['device']

    def __init__(self, config_file):
        config = toml.load(config_file)
        self.data = self.Data(config['data'])
        self.model = self.Model(config['model'])
        self.device = self.Device(config['device'])
