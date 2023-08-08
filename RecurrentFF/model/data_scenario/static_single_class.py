import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import RMSprop
import wandb

from RecurrentFF.model.data_scenario.processor import DataScenarioProcessor
from RecurrentFF.model.inner_layers import InnerLayers
from RecurrentFF.util import LatentAverager, TrainLabelData, layer_activations_to_goodness, ForwardMode
from RecurrentFF.settings import Settings, DataConfig


class SingleStaticClassTestData:
    """
    inputs of dims (timesteps, batch_size, data_size)
    labels of dims (batch size, num classes)
    """

    def __init__(self, input, labels):
        self.input = input
        self.labels = labels

    def __iter__(self):
        yield self.input
        yield self.labels


def formulate_incorrect_class(prob_tensor: torch.Tensor,
                              correct_onehot_tensor: torch.Tensor,
                              settings: Settings) -> torch.Tensor:
    # Compute the indices of the correct class for each sample
    correct_indices = correct_onehot_tensor.argmax(dim=1)

    # Compute the indices of the maximum probability for each sample
    max_prob_indices = prob_tensor.argmax(dim=1)

    # Compute the percentage where the maximum probability index matches the
    # correct class index
    percentage_matching = (
        max_prob_indices == correct_indices).float().mean().item() * 100
    logging.info(
        f"Latent classifier accuracy: {percentage_matching}%")

    if settings.model.should_log_metrics:
        wandb.log({
            "latent_classifier_acc": percentage_matching
        })

    # Extract the probabilities of the correct classes
    correct_probs = prob_tensor.gather(
        1, correct_indices.unsqueeze(1)).squeeze()

    # Generate random numbers for each sample in the range [0, 1]
    rand_nums = torch.rand_like(correct_probs).unsqueeze(
        1).to(device=settings.device.device)

    # Zero out the probabilities corresponding to the correct class
    # Make a copy to avoid in-place modifications
    masked_prob_tensor = prob_tensor.clone() + settings.model.epsilon
    masked_prob_tensor.scatter_(1, correct_indices.unsqueeze(1), 0)

    # Normalize the masked probabilities such that they sum to 1 along the
    # class dimension
    normalized_masked_prob_tensor = masked_prob_tensor / \
        masked_prob_tensor.sum(dim=1, keepdim=True)

    # Create a cumulative sum of the masked probabilities along the classes
    # dimension
    cumulative_prob = torch.cumsum(normalized_masked_prob_tensor, dim=1)

    # Expand random numbers to the same shape as cumulative_prob for comparison
    rand_nums_expanded = rand_nums.expand_as(cumulative_prob)

    # Create a mask that identifies where the random numbers are less than the
    # cumulative probabilities
    mask = (rand_nums_expanded < cumulative_prob).int()

    # Use argmax() to find the index of the first True value in each row
    selected_indices = mask.argmax(dim=1)

    # Create a tensor with zeros and the same shape as the prob_tensor
    result_onehot_tensor = torch.zeros_like(
        prob_tensor).to(device=settings.device.device)

    # Batch-wise assignment of 1 to the selected indices
    result_onehot_tensor.scatter_(1, selected_indices.unsqueeze(1), 1)

    # Compute accuracy
    max_indices_correct = correct_onehot_tensor.argmax(dim=1)
    correct = (selected_indices == max_indices_correct).sum().item()
    incorrect = prob_tensor.size(0) - correct

    logging.info("Optimization classifier accuracy: " +
                 str(correct / (correct + incorrect)))

    return result_onehot_tensor


class StaticSingleClassProcessor(DataScenarioProcessor):
    def __init__(self, inner_layers: InnerLayers, settings: Settings):
        self.settings = settings
        self.inner_layers = inner_layers

        self.classification_weights = nn.Linear(
            sum(
                self.settings.model.hidden_sizes),
            self.settings.data_config.num_classes).to(
            device=self.settings.device.device)

        if self.settings.model.classifier_optimizer == "rmsprop":
            self.optimizer = RMSprop(
                self.classification_weights.parameters(),
                momentum=self.settings.model.classifier_rmsprop.momentum,
                lr=self.settings.model.classifier_rmsprop.learning_rate)
        elif self.settings.model.classifier_optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.classification_weights.parameters(),
                lr=self.settings.model.classifier_adam.learning_rate)

    def train_class_predictor_from_latents(
            self, latents: torch.Tensor, labels: torch.Tensor):
        """
        Trains the classification model using the given latent vectors and
        corresponding labels.

        The method performs one step of optimization by computing the cross-
        entropy loss between the predicted logits and the true labels,
        performing backpropagation, and then updating the model's parameters.

        Args:
            latents (torch.Tensor): A tensor containing the latent
                representations of the inputs.
            labels (torch.Tensor): A tensor containing the true labels
                corresponding to the latents.
        """
        self.optimizer.zero_grad()
        latents = latents.detach()

        class_logits = F.linear(
            latents, self.classification_weights.weight)

        loss = F.cross_entropy(
            class_logits, labels)

        if self.settings.model.should_log_metrics:
            wandb.log({
                "latent_classifier_loss": loss
            })

        loss.backward()

        self.optimizer.step()

        logging.info(
            f"loss for training optimization classifier: {loss.item()}")

    def replace_negative_data_inplace(
            self,
            input_batch: torch.Tensor,
            input_labels: TrainLabelData):
        """
        Replaces the negative labels in the given input labels with incorrect
        class labels, based on the latent representations of the input batch.

        This method retrieves the latents, computes the class logits and
        probabilities, and then formulates incorrect class labels, replacing
        the negative labels in the input labels in-place.

        The point is to choose samples which the model thinks are positive data,
        but aren't.

        Args:
            input_batch (torch.Tensor): A tensor containing the input batch
                of data.
            input_labels (TrainLabelData): A custom data structure containing
                positive and negative labels, where the negative labels will
                be replaced.
        """
        latents = self.__retrieve_latents__(input_batch, input_labels)

        print(latents.shape)
        print(self.classification_weights.weight.shape)

        class_logits = F.linear(latents, self.classification_weights.weight)
        class_probabilities = F.softmax(class_logits, dim=-1)
        negative_labels = formulate_incorrect_class(
            class_probabilities, input_labels.pos_labels[0], self.settings)

        frames = input_labels.pos_labels.shape[0]
        negative_labels = negative_labels.unsqueeze(
            0)  # Add a new dimension at the beginning
        input_labels.neg_labels = negative_labels.repeat(
            frames, 1, 1)  # Repeat along the new dimension

    def brute_force_predict(self, test_loader, limit_batches=None):
        """
        This function predicts the class labels for the provided test data using
        the trained RecurrentFFNet model. It does so by enumerating all possible
        class labels and choosing the one that produces the highest 'goodness'
        score. We cannot use this function for datasets with changing classes.

        Args:
            test_data (object): A tuple containing the test data, one-hot labels,
            and the actual labels. The data is assumed to be PyTorch tensors.

        Returns:
            float: The prediction accuracy as a percentage.

        Procedure:
            The function first moves the test data and labels to the appropriate
            device. It then calculates the 'goodness' metric for each possible
            class label, using a two-step process:

                1. Resetting the network's activations and forwarding the data
                   through the network with the current label.
                2. For each iteration within a specified threshold, forwarding
                   the data again, but this time retaining the
                activations, which are used to calculate the 'goodness' for each
                layer.

            The 'goodness' values across iterations and layers are then averaged
            to produce a single 'goodness' score for each class label. The class
            with the highest 'goodness' score is chosen as the prediction for
            each test sample.

            Finally, the function calculates the overall accuracy of the model's
            predictions by comparing them to the actual labels and returns this
            accuracy.
        """
        for batch, test_data in enumerate(test_loader):
            if limit_batches is not None and batch == limit_batches:
                break

            # tuple: (correct, total)
            accuracy_contexts = []

            with torch.no_grad():
                data, labels = test_data
                data = data.to(self.settings.device.device)
                labels = labels.to(self.settings.device.device)

                iterations = data.shape[0]

                all_labels_goodness = []

                # evaluate goodness for each possible label
                for label in range(self.settings.data_config.num_classes):
                    self.inner_layers.reset_activations(False)

                    one_hot_labels = torch.zeros(
                        data.shape[1],
                        self.settings.data_config.num_classes,
                        device=self.settings.device.device)
                    one_hot_labels[:, label] = 1.0

                    for _preinit_iteration in range(0, len(self.inner_layers)):
                        self.inner_layers.advance_layers_forward(
                            ForwardMode.PredictData, data[0], one_hot_labels, False)

                    lower_iteration_threshold = iterations // 2 - \
                        iterations // 10
                    upper_iteration_threshold = iterations // 2 + \
                        iterations // 10
                    goodnesses = []
                    for iteration in range(0, iterations):
                        self.inner_layers.advance_layers_forward(
                            ForwardMode.PredictData, data[iteration], one_hot_labels, True)

                        if iteration >= lower_iteration_threshold and iteration <= upper_iteration_threshold:
                            layer_goodnesses = []
                            for layer in self.inner_layers:
                                layer_goodnesses.append(
                                    layer_activations_to_goodness(
                                        layer.predict_activations.current))

                            goodnesses.append(torch.stack(
                                layer_goodnesses, dim=1))

                    # tensor of shape (batch_size, iterations, num_layers)
                    goodnesses = torch.stack(goodnesses, dim=1)
                    # average over iterations
                    goodnesses = goodnesses.mean(dim=1)
                    # average over layers
                    goodness = goodnesses.mean(dim=1)

                    logging.debug("Goodness for prediction" + " " +
                                  str(label) + ": " + str(goodness))
                    all_labels_goodness.append(goodness)

                all_labels_goodness = torch.stack(all_labels_goodness, dim=1)

                # select the label with the maximum goodness
                predicted_labels = torch.argmax(all_labels_goodness, dim=1)
                logging.debug("Predicted labels: " + str(predicted_labels))
                logging.debug("Actual labels: " + str(labels))

                total = data.size(1)
                correct = (predicted_labels == labels).sum().item()

                accuracy_contexts.append((correct, total))

        total_correct = sum(correct for correct, _total in accuracy_contexts)
        total_submissions = sum(
            total for _correct, total in accuracy_contexts)
        accuracy = total_correct / total_submissions * 100 if total_submissions else 0
        logging.info(f'Test accuracy: {accuracy}%')

        return accuracy

    def __retrieve_latents__(
            self,
            input_batch: torch.Tensor,
            input_labels: TrainLabelData) -> torch.Tensor:
        self.inner_layers.reset_activations(True)

        # assign equal probability to all labels
        batch_size = input_labels.pos_labels[0].shape[0]
        num_classes = input_labels.pos_labels[0].shape[1]
        equally_distributed_class_labels = torch.full(
            (batch_size,
             num_classes),
            1 /
            num_classes).to(
            device=self.settings.device.device)

        iterations = input_batch.shape[0]

        # feed data through network and track latents
        for _preinit_iteration in range(0, len(self.inner_layers)):
            self.inner_layers.advance_layers_forward(
                ForwardMode.PositiveData,
                input_batch[0],
                equally_distributed_class_labels,
                False)

        lower_iteration_threshold = iterations // 2 - \
            iterations // 10
        upper_iteration_threshold = iterations // 2 + \
            iterations // 10

        target_latents = LatentAverager()
        for iteration in range(0, iterations):
            self.inner_layers.advance_layers_forward(
                ForwardMode.PositiveData,
                input_batch[iteration],
                equally_distributed_class_labels,
                True)

            if iteration >= lower_iteration_threshold and iteration <= upper_iteration_threshold:
                latents = [
                    layer.pos_activations.current for layer in self.inner_layers]
                latents = torch.cat(latents, dim=1).to(
                    device=self.settings.device.device)
                target_latents.track_collapsed_latents(latents)

        return target_latents.retrieve()
