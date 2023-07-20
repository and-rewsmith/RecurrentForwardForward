import torch

# TODO: expose as hyperparams
EPOCHS = 1000000
THRESHOLD = 1
DAMPING_FACTOR = 0.7
EPSILON = 1e-8
LEARNING_RATE = 0.00005
DEVICE = torch.device("cuda")
