from typing_extensions import Self
import toml

from pydantic import BaseModel

from RecurrentFF.benchmarks.arguments import get_arguments

CONFIG_FILE = "./config.toml"


class DataConfig(BaseModel):
    data_size: int
    num_classes: int
    train_batch_size: int
    test_batch_size: int
    iterations: int
    dataset: str


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


class FfAdadelta(BaseModel):
    learning_rate: float


class ClassifierAdadelta(BaseModel):
    learning_rate: float


class Model(BaseModel):
    hidden_sizes: list
    epochs: int
    prelabel_timesteps: int
    loss_threshold: float
    damping_factor: float
    epsilon: float
    skip_profiling: bool
    should_log_metrics: bool
    should_replace_neg_data: bool
    should_load_weights: bool

    ff_activation: str

    ff_optimizer: str
    ff_rmsprop: FfRmsprop
    ff_adam: FfAdam
    ff_adadelta: FfAdadelta

    classifier_optimizer: str
    classifier_rmsprop: ClassifierRmsprop
    classifier_adam: ClassifierAdam
    classifier_adadelta: FfAdadelta


class Device(BaseModel):
    device: str


class Settings(BaseModel):
    model: Model
    device: Device

    # needs to be None because it is not in the config file
    data_config: DataConfig = None  # type: ignore[assignment]

    @classmethod
    def from_config_file(cls, path: str) -> Self:
        config = toml.load(path)
        model = config['model']

        if model['ff_optimizer'] == "rmsprop":
            model['ff_rmsprop'] = FfRmsprop(**model['ff_rmsprop'])
        elif model['ff_optimizer'] == "adam":
            model['ff_adam'] = FfAdam(**model['ff_adam'])
        elif model['ff_optimizer'] == "adadelta":
            model['ff_adadelta'] = FfAdadelta(**model['ff_adadelta'])

        if model['classifier_optimizer'] == "rmsprop":
            model['classifier_rmsprop'] = ClassifierRmsprop(
                **model['classifier_rmsprop'])
        elif model['classifier_optimizer'] == "adam":
            model['classifier_adam'] = ClassifierAdam(
                **model['classifier_adam'])
        elif model['classifier_optimizer'] == "adadelta":
            model['classifier_adadelta'] = FfAdadelta(
                **model['classifier_adadelta'])

        if "data_config" in config:
            data_config = DataConfig(**config["data_config"])
            return cls(
                model=Model(
                    **model),
                device=Device(
                    **config['device']),
                data_config=data_config)
        else:
            return cls(model=Model(**model), device=Device(**config['device']))

    @classmethod
    def new(cls) -> Self:
        args = get_arguments()
        config_file = args.config_file if args.config_file is not None else CONFIG_FILE
        return cls.from_config_file(config_file)
