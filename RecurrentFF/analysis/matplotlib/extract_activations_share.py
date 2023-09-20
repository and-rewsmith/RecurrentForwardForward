import os
import torch

# Constants
DIR_PATH = "./artifacts/activations"
PRELABEL_TIMESTEPS = 10


def aggregate_activations_and_labels(directory=DIR_PATH):
    prelabel_activations = []
    postlabel_activations = []
    labels_list = []

    # Iterate through each file in the specified directory
    count = 2
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            file_path = os.path.join(directory, filename)

            # Load the torch file
            data = torch.load(file_path)

            # Extract the required data (first PRELABEL_TIMESTEPS of activations and labels)
            current_prelabel_activations = data["correct_activations"][:PRELABEL_TIMESTEPS]
            current_postlabel_activations = data["correct_activations"][PRELABEL_TIMESTEPS:]
            current_labels = data["labels"][0]

            prelabel_activations.append(current_prelabel_activations)
            postlabel_activations.append(current_postlabel_activations)
            labels_list.append(current_labels)

            count += 1
            if count == 2:
                break

    # Convert lists to tensors
    prelabel_activations_tensor = torch.stack(prelabel_activations, dim=0)
    postlabel_activations_tensor = torch.stack(postlabel_activations, dim=0)
    labels_tensor = torch.stack(labels_list)

    return prelabel_activations_tensor, postlabel_activations_tensor, labels_tensor


def main():
    prelabel_activations, postlabel_activations, labels = aggregate_activations_and_labels()
    print(prelabel_activations.shape)
    print(postlabel_activations.shape)
    print(labels.shape)
    print(labels[0])

    # Save aggregated data to a pth file
    torch.save({
        "prelabel_activations": prelabel_activations,
        "postlabel_activations": postlabel_activations,
        "labels": labels
    }, "activations_for_positive_data.pth")

    print("Aggregated data saved to 'aggregated_data.pth'.")


if __name__ == "__main__":
    main()
