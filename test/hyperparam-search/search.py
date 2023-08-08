import random

from RecurrentFF.settings import Settings


settings = Settings.from_config_file("./test/config-files/smoke.toml")
print(settings.model_dump())

EPOCHS = 1

hidden_sizes = [[1500, 1500, 1500], [2000, 2000, 2000],
                [2500, 2500, 2500], [2000, 2000, 2000, 2000]]

ff_act = ["relu", "leaky_relu"]

ff_optimizers = ["rmsprop", "adam"]
classifier_optimizers = ["rmsprop", "adam"]

ff_rmsprop_momentums = [0.0, 0.2, 0.5, 0.9]
ff_rmsprop_learning_rates = [0.00001, 0.0001, 0.001]

classifier_rmsprop_momentums = [0.0, 0.2, 0.5, 0.9]
classifier_rmsprop_learning_rates = [0.0001, 0.001, 0.01]

ff_adam_learning_rates = [0.00001, 0.0001, 0.001]

classifier_adam_learning_rates = [0.0001, 0.001, 0.01]

seen = set()

while True:
    # generate random entry from ff_act
    act = random.choice(ff_act)
    ff_opt = random.choice(ff_optimizers)
    classifier_opt = random.choice(classifier_optimizers)
    ff_rmsprop_momentum = random.choice(ff_rmsprop_momentums)
    ff_rmsprop_learning_rate = random.choice(ff_rmsprop_learning_rates)
    classifier_rmsprop_momentum = random.choice(
        classifier_rmsprop_momentums)
    classifier_rmsprop_learning_rate = random.choice(
        classifier_rmsprop_learning_rates)
    ff_adam_learning_rate = random.choice(ff_adam_learning_rates)
    classifier_adam_learning_rate = random.choice(
        classifier_adam_learning_rates)

    entry = str(
        hidden_sizes) + "," + str(act) + "," + str(ff_opt) + "," + str(classifier_opt)

    if ff_opt == "rmsprop":
        entry += "," + str(ff_rmsprop_learning_rate) + "," + \
            str(ff_rmsprop_momentum)
    elif ff_opt == "adam":
        entry += "," + \
            str(ff_adam_learning_rate)

    if classifier_opt == "rmsprop":
        entry += "," + str(classifier_rmsprop_learning_rate) + "," + \
            str(classifier_rmsprop_momentum)
    elif classifier_opt == "adam":
        entry += "," + \
            str(classifier_adam_learning_rate)

    if entry not in seen:
        print(entry)

    seen.add(entry)
