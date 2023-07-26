import logging
import os
import string
import random

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor

from constants import MOVING_MNIST_DATA_DIR


BATCH_SIZE = 1


def get_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters_and_digits)
                         for _ in range(length))
    return result_str


def move_images(images, frames, velocities):
    # Expand canvas from 28x28 to 64x64
    extended_images = torch.zeros((images.shape[0], 64, 64))
    extended_images[:, 18:46, 18:46] = images

    # Initialize sequence tensor
    sequence = torch.zeros((images.shape[0], frames, 64, 64))

    # Set initial positions
    # Starting at the center
    positions = torch.full((images.shape[0], 2), 18)

    for frame in range(frames):
        # Clear old positions
        extended_images *= 0
        # Apply new positions
        for i, (x, y) in enumerate(positions):
            extended_images[i, x:x+28, y:y+28] = images[i]

        sequence[:, frame] = extended_images

        # Update positions
        positions += velocities

        # Reflect pixels off boundaries
        bounce = ((positions < 0) | (positions > 36))
        velocities[bounce] *= -1
        positions[positions < 0] = 0
        positions[positions > 36] = 36

    return sequence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    transform = Compose([
        ToTensor(),
    ])

    # ===========================TRAIN===========================

    mnist_loader = DataLoader(MNIST('./data/', train=True, download=True, transform=transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # create directory for data if it doesn't exist
    os.makedirs(MOVING_MNIST_DATA_DIR, exist_ok=True)

    # Test on each batch
    sequences_build = []
    labels_build = []
    for i, (images, labels) in enumerate(mnist_loader):
        images = images.squeeze(1)  # Remove channel dimension for simplicity
        # Random velocities
        velocities = torch.randint(-5, 6, (images.shape[0], 2))
        sequences = move_images(images, 50, velocities)

        sequences_build.append(sequences)
        labels_build.append(labels)

        if i != 0 and i % 1000 == 0:
            data = {'sequences': torch.cat(
                sequences_build, dim=0), 'labels': torch.cat(labels_build, dim=0)}
            torch.save(data, f'{MOVING_MNIST_DATA_DIR}/train_{i}.pt')
            sequences_build.clear()
            labels_build.clear()
            print(f"Saved {i} train sequences")
        elif i == len(mnist_loader) - 1:
            data = {'sequences': torch.cat(
                sequences_build, dim=0), 'labels': torch.cat(labels_build, dim=0)}
            torch.save(
                data, f'{MOVING_MNIST_DATA_DIR}/train_{len(mnist_loader)}.pt')

    # ===========================TEST===========================

    mnist_loader = DataLoader(MNIST('./data/', train=False, download=True, transform=transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Test on each batch
    sequences_build = []
    labels_build = []
    for i, (images, labels) in enumerate(mnist_loader):
        images = images.squeeze(1)  # Remove channel dimension for simplicity
        # Random velocities
        velocities = torch.randint(-5, 6, (images.shape[0], 2))
        sequences = move_images(images, 50, velocities)

        sequences_build.append(sequences)
        labels_build.append(labels)

        if i != 0 and i % 1000 == 0:
            data = {'sequences': torch.cat(
                sequences_build, dim=0), 'labels': torch.cat(labels_build, dim=0)}
            torch.save(data, f'{MOVING_MNIST_DATA_DIR}/test_{i}.pt')
            sequences_build.clear()
            labels_build.clear()
            print(f"Saved {i} test sequences")
        elif i == len(mnist_loader) - 1:
            data = {'sequences': torch.cat(
                sequences_build, dim=0), 'labels': torch.cat(labels_build, dim=0)}
            torch.save(
                data, f'{MOVING_MNIST_DATA_DIR}/test_{len(mnist_loader)}.pt')
