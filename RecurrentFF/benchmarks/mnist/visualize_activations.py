import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

tensors = torch.load('test_sample_1.pt')

correct = tensors['incorrect_activations']
print(correct.shape)

for i in range(0, correct.shape[0]):
    print(correct[i].shape)
    input()
    df = pd.DataFrame(correct[i].cpu().numpy())
    plt.figure(figsize=(10, 5))
    sns.heatmap(df, cmap='viridis')
    plt.title('Activation Heatmap')
    plt.xlabel('Neuron #')
    plt.ylabel('Layers')
    plt.show()
