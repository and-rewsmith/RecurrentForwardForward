import torch
import pandas as pd
import os

# Define constants
DATASET = "MNIST"
LABEL_SHOW_BOUNDARY = 10
NUMBER_FILES = 100
DIR_PATH = "artifacts/activations"
DEVICE = "cuda"
BATCH_SIZE = 1024 ** 3

# Get the first NUMBER_FILES files from the directory
files = [f for f in os.listdir(DIR_PATH) if f.endswith('.pt')]
files = sorted(files)[:NUMBER_FILES]

df = pd.DataFrame()
batch_rows = []

for i, file in enumerate(files):
    print("processing file", i, "of", len(files))

    # Load data
    data = torch.load(os.path.join(DIR_PATH, file), map_location=DEVICE)

    for data_type, label in [("correct", 1), ("incorrect", 0)]:
        for timestep, _ in enumerate(data[f"{data_type}_activations"]):
            is_label_showing = 1 if timestep >= LABEL_SHOW_BOUNDARY else 0

            print("processing timestep ", timestep)

            for layer_index, _ in enumerate(
                    data[f"{data_type}_activations"][timestep]):
                for neuron_index, _ in enumerate(
                        data[f"{data_type}_activations"][timestep][layer_index]):
                    row = {
                        "image_timestep": timestep,
                        "layer_index": layer_index,
                        "is_label_showing": is_label_showing,
                        "neuron index": neuron_index,
                        "image": ','.join(map(str, data['data'][timestep].cpu().numpy().flatten())),
                        "label": data['labels'][0].cpu().numpy().flatten(),
                        "dataset": DATASET,
                        "is_correct": True if data_type == "correct" else False,
                        "data_sample_id": i,
                        "activation":
                            data[f"{data_type}_activations"][timestep][layer_index][neuron_index].cpu().item(),
                        "forward_activation_component":
                            data[f"{data_type}_forward_activations"][timestep][layer_index][neuron_index].cpu().item(),
                        "backward_activation_component":
                            data[f"{data_type}_backward_activations"][timestep][layer_index][neuron_index].cpu().item(),
                        "lateral_activation_component":
                            data[f"{data_type}_lateral_activations"][timestep][layer_index][neuron_index].cpu().item()
                    }
                    batch_rows.append(row)

                    if len(batch_rows) == BATCH_SIZE:
                        df = pd.concat(
                            [df, pd.DataFrame(batch_rows)], ignore_index=True)
                        batch_rows.clear()
                        df.to_parquet('converted_data.parquet')
                        print("saved batch to parquet file")

if batch_rows:
    df = pd.concat([df, pd.DataFrame(batch_rows)], ignore_index=True)

df.to_parquet('converted_data.parquet')

print("Data conversion complete.")
