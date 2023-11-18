import logging
from typing import List, Optional, cast
from pyparsing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import RMSprop, Optimizer
import wandb

from RecurrentFF.model.data_scenario.processor import DataScenarioProcessor
from RecurrentFF.model.inner_layers import InnerLayers
from RecurrentFF.util import Activations, LatentAverager, TrainLabelData, layer_activations_to_badness, ForwardMode, scale_labels_by_timestep
from RecurrentFF.settings import Settings


class SingleStaticClassTestData:
    """
    inputs of dims (timesteps, batch_size, data_size)
    labels of dims (batch size, num classes)
    """

    def __init__(self, input: torch.Tensor, labels: torch.Tensor) -> None:
        self.input = input
        self.labels = labels

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.input
        yield self.labels


class StaticSingleClassActivityTracker():

    def __init__(self) -> None:
        self.data: Optional[torch.Tensor]
        self.labels: Optional[torch.Tensor]

        self.activations: List[torch.Tensor] = []
        self.forward_activations: List[torch.Tensor] = []
        self.backward_activations: List[torch.Tensor] = []
        self.lateral_activations: List[torch.Tensor] = []

        self.partial_activations: List[torch.Tensor] = []
        self.partial_forward_activations: List[torch.Tensor] = []
        self.partial_backward_activations: List[torch.Tensor] = []
        self.partial_lateral_activations: List[torch.Tensor] = []

        self.tracked_samples = 0

    def reinitialize(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data[0][0]  # first batch, first timestep
        self.labels = labels.squeeze(1)

        self.activations = []
        self.forward_activations = []
        self.backward_activations = []
        self.lateral_activations = []

        self.partial_activations = []
        self.partial_forward_activations = []
        self.partial_backward_activations = []
        self.partial_lateral_activations = []

        self.tracked_samples += 1

    def track_partial_activations(self, layers: InnerLayers) -> None:
        build = []
        build_forward = []
        build_backward = []
        build_lateral = []
        for layer in layers:
            # TODO: cleanup cast
            build.append(cast(Activations, layer.predict_activations).current)
            build_forward.append(layer.forward_act)
            build_backward.append(layer.backward_act)
            build_lateral.append(layer.lateral_act)

        self.partial_activations.append(torch.stack(build).squeeze(1))
        self.partial_forward_activations.append(
            torch.stack(build_forward).squeeze(1))
        self.partial_backward_activations.append(
            torch.stack(build_backward).squeeze(1))
        self.partial_lateral_activations.append(
            torch.stack(build_lateral).squeeze(1))

    def cut_activations(self) -> None:
        self.activations.append(torch.stack(self.partial_activations))
        self.forward_activations.append(
            torch.stack(self.partial_forward_activations))
        self.backward_activations.append(
            torch.stack(self.partial_backward_activations))
        self.lateral_activations.append(
            torch.stack(self.partial_lateral_activations))

        self.partial_activations = []
        self.partial_forward_activations = []
        self.partial_backward_activations = []
        self.partial_lateral_activations = []

    def filter_and_persist(
            self,
            predicted_labels: torch.Tensor,
            anti_predictions: torch.Tensor,
            actual_labels: torch.Tensor) -> None:
        assert self.data is not None and self.labels is not None

        if predicted_labels == actual_labels:
            predicted_labels_index = int(predicted_labels.item())
            anti_prediction_index = int(anti_predictions.item())

            correct_activations = self.activations[predicted_labels_index]
            correct_forward_activations = self.forward_activations[predicted_labels_index]
            correct_backward_activations = self.backward_activations[predicted_labels_index]
            correct_lateral_activations = self.lateral_activations[predicted_labels_index]

            incorrect_activations = self.activations[anti_prediction_index]
            incorrect_forward_activations = self.forward_activations[anti_prediction_index]
            incorrect_backward_activations = self.backward_activations[anti_prediction_index]
            incorrect_lateral_activations = self.lateral_activations[anti_prediction_index]

            logging.debug(f"Correct activations: {correct_activations.shape}")
            logging.debug(
                f"Incorrect activations: {incorrect_activations.shape}")
            logging.debug(f"Data: {self.data.shape}")
            logging.debug(f"Labels: {self.labels.shape}")

            torch.save({
                "correct_activations": correct_activations,
                "correct_forward_activations": correct_forward_activations,
                "correct_backward_activations": correct_backward_activations,
                "correct_lateral_activations": correct_lateral_activations,
                "incorrect_activations": incorrect_activations,
                "incorrect_forward_activations": incorrect_forward_activations,
                "incorrect_backward_activations": incorrect_backward_activations,
                "incorrect_lateral_activations": incorrect_lateral_activations,
                "data": self.data,
                "labels": self.labels
            },
                f"artifacts/activations/test_sample_{self.tracked_samples}.pt")

        else:
            logging.warn("Predicted label does not match actual label")
            self.activations = []
            self.forward_activations = []
            self.backward_activations = []
            self.lateral_activations = []

            self.partial_activations = []
            self.partial_forward_activations = []
            self.partial_backward_activations = []
            self.partial_lateral_activations = []

            self.data = None
            self.labels = None


def formulate_incorrect_class(prob_tensor: torch.Tensor,
                              correct_onehot_tensor: torch.Tensor,
                              settings: Settings,
                              total_batch_count: int) -> torch.Tensor:
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
        }, step=total_batch_count)

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
    def __init__(self, inner_layers: InnerLayers, settings: Settings, noise: torch.Tensor):
        self.settings = settings
        self.inner_layers = inner_layers

        self.noise = noise

        # pytorch types are incorrect hence ignore statement
        self.classification_weights = nn.Linear(  # type: ignore[call-overload]
            sum(
                self.settings.model.hidden_sizes),
            self.settings.data_config.num_classes).to(
            device=self.settings.device.device)

        self.optimizer: Optimizer

        if self.settings.model.classifier_optimizer == "rmsprop":
            self.optimizer = RMSprop(
                self.classification_weights.parameters(),
                momentum=self.settings.model.classifier_rmsprop.momentum,
                lr=self.settings.model.classifier_rmsprop.learning_rate)
        elif self.settings.model.classifier_optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.classification_weights.parameters(),
                lr=self.settings.model.classifier_adam.learning_rate)
        elif self.settings.model.classifier_optimizer == "adadelta":
            self.optimizer = torch.optim.Adadelta(
                self.classification_weights.parameters(),
                lr=self.settings.model.classifier_adadelta.learning_rate)

    def train_class_predictor_from_latents(
            self,
            latents: torch.Tensor,
            labels: torch.Tensor,
            total_batch_count: int) -> None:
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
            }, step=total_batch_count)

        loss.backward()

        self.optimizer.step()

        logging.info(
            f"loss for training optimization classifier: {loss.item()}")

    def replace_negative_data_inplace(
            self,
            input_batch: torch.Tensor,
            input_labels: TrainLabelData,
            total_batch_count: int) -> None:
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

        class_logits = F.linear(latents, self.classification_weights.weight)
        class_probabilities = F.softmax(class_logits, dim=-1)
        negative_labels = formulate_incorrect_class(
            class_probabilities,
            input_labels.pos_labels[0],
            self.settings,
            total_batch_count)

        frames = input_labels.pos_labels.shape[0]
        negative_labels = negative_labels.unsqueeze(
            0)  # Add a new dimension at the beginning
        input_labels.neg_labels = negative_labels.repeat(
            frames, 1, 1)  # Repeat along the new dimension

    def brute_force_predict(
            self,
            loader: torch.utils.data.DataLoader,
            limit_batches: Optional[int] = None,
            is_test_set: bool = False,
            write_activations: bool = False) -> float:
        """
        This function predicts the class labels for the provided test data using
        the trained RecurrentFFNet model. It does so by enumerating all possible
        class labels and choosing the one that produces the lowest 'badness'
        score. We cannot use this function for datasets with changing classes.

        Args:
            test_data (object): A tuple containing the test data, one-hot labels,
            and the actual labels. The data is assumed to be PyTorch tensors.

        Returns:
            float: The prediction accuracy as a percentage.

        Procedure:
            The function first moves the test data and labels to the appropriate
            device. It then calculates the 'badness' metric for each possible
            class label, using a two-step process:

                1. Resetting the network's activations and forwarding the data
                   through the network with the current label.
                2. For each iteration within a specified threshold, forwarding
                   the data again, but this time retaining the
                activations, which are used to calculate the 'badness' for each
                layer.

            The 'badness' values across iterations and layers are then averaged
            to produce a single 'badness' score for each class label. The class
            with the lowest 'badness' score is chosen as the prediction for
            each test sample.

            Finally, the function calculates the overall accuracy of the model's
            predictions by comparing them to the actual labels and returns this
            accuracy.
        """
        if write_activations:
            assert self.settings.data_config.test_batch_size == 1 \
                and is_test_set, "Cannot write activations for batch size > 1"
            activity_tracker = StaticSingleClassActivityTracker()

        forward_mode = ForwardMode.PredictData if is_test_set else ForwardMode.PositiveData

        # tuple: (correct, total)
        accuracy_contexts = []

        for batch, test_data in enumerate(loader):

            if limit_batches is not None and batch == limit_batches:
                break

            with torch.no_grad():
                data, labels = test_data
                data = data.to(self.settings.device.device)
                labels = labels.to(self.settings.device.device)

                batch_size_for_noise = self.settings.data_config.test_batch_size if is_test_set else self.settings.data_config.train_batch_size

                broadcasted_noise = self.noise[0].unsqueeze(0).unsqueeze(0).expand(
                    self.settings.data_config.iterations, batch_size_for_noise, -1)

                data += broadcasted_noise

                if write_activations:
                    activity_tracker.reinitialize(data, labels)

                # since this is static singleclass we can use the first frame
                # for the label
                labels = labels[0]

                iterations = data.shape[0]

                all_labels_badness = []

                # evaluate badness for each possible label
                for label in range(self.settings.data_config.num_classes):
                    self.inner_layers.reset_activations(not is_test_set)

                    upper_clamped_tensor = self.get_preinit_upper_clamped_tensor(
                        (data.shape[1], self.settings.data_config.num_classes))

                    for _preinit_step in range(
                            0, self.settings.model.prelabel_timesteps):
                        self.inner_layers.advance_layers_forward(
                            forward_mode, data[0], upper_clamped_tensor, False)
                        if write_activations:
                            activity_tracker.track_partial_activations(
                                self.inner_layers)

                    one_hot_labels = torch.zeros(
                        data.shape[1],
                        self.settings.data_config.num_classes,
                        device=self.settings.device.device)
                    one_hot_labels[:, label] = 1.0

                    lower_iteration_threshold = iterations // 2 - \
                        iterations // 10
                    upper_iteration_threshold = iterations // 2 + \
                        iterations // 10
                    badnesses = []
                    for iteration in range(0, iterations):
                        # one_hot_labels_rescaled = scale_labels_by_timestep(
                        #     one_hot_labels, iteration, iterations)

                        self.inner_layers.advance_layers_forward(
                            forward_mode, data[iteration], one_hot_labels, True)

                        if write_activations:
                            activity_tracker.track_partial_activations(
                                self.inner_layers)

                        if iteration >= lower_iteration_threshold and iteration <= upper_iteration_threshold:
                            layer_badnesses = []
                            for layer in self.inner_layers:
                                activations = cast(Activations, layer.pos_activations).current \
                                    if forward_mode == ForwardMode.PositiveData \
                                    else cast(Activations, layer.predict_activations).current

                                layer_badnesses.append(
                                    layer_activations_to_badness(
                                        activations))

                            badnesses.append(torch.stack(
                                layer_badnesses, dim=1))

                    if write_activations:
                        activity_tracker.cut_activations()

                    # tensor of shape (batch_size, iterations, num_layers)
                    badnesses_stacked = torch.stack(badnesses, dim=1)
                    badnesses_mean_over_iterations = badnesses_stacked.mean(
                        dim=1)
                    badness_mean_over_layers = badnesses_mean_over_iterations.mean(
                        dim=1)

                    logging.debug("Badness for prediction" + " " +
                                  str(label) + ": " + str(badness_mean_over_layers))
                    all_labels_badness.append(badness_mean_over_layers)

                all_labels_badness_stacked = torch.stack(
                    all_labels_badness, dim=1)

                # select the label with the maximum badness
                predicted_labels = torch.argmin(
                    all_labels_badness_stacked, dim=1)
                if write_activations:
                    anti_predictions = torch.argmax(
                        all_labels_badness_stacked, dim=1)
                    activity_tracker.filter_and_persist(
                        predicted_labels, anti_predictions, labels)

                logging.debug("Predicted labels: " + str(predicted_labels))
                logging.debug("Actual labels: " + str(labels))

                total = data.size(1)
                correct = (predicted_labels == labels).sum().item()

                accuracy_contexts.append((correct, total))

        total_correct = sum(correct for correct, _total in accuracy_contexts)
        total_submissions = sum(
            total for _correct, total in accuracy_contexts)
        accuracy: float = total_correct / total_submissions * \
            100

        if is_test_set:
            logging.info(f'Test accuracy: {accuracy}%')
        else:
            logging.info(f'Train accuracy: {accuracy}%')

        return accuracy

    def get_preinit_upper_clamped_tensor(
            self, upper_clamped_tensor_shape: tuple) -> torch.Tensor:
        labels = torch.full(
            upper_clamped_tensor_shape,
            1.0 / self.settings.data_config.num_classes,
            device=self.settings.device.device)
        return labels

    def __retrieve_latents__(
            self,
            input_batch: torch.Tensor,
            input_labels: TrainLabelData) -> torch.Tensor:
        self.inner_layers.reset_activations(True)

        # assign equal probability to all labels
        batch_size = input_labels.pos_labels[0].shape[0]
        equally_distributed_class_labels = torch.full(
            (batch_size,
             self.settings.data_config.num_classes),
            1 /
            self.settings.data_config.num_classes).to(
            device=self.settings.device.device)

        iterations = input_batch.shape[0]

        # feed data through network and track latents
        for _preinit_step in range(0, self.settings.model.prelabel_timesteps):
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
                    cast(Activations, layer.pos_activations).current for layer in self.inner_layers]
                latents_collapsed = torch.cat(latents, dim=1).to(
                    device=self.settings.device.device)
                target_latents.track_collapsed_latents(latents_collapsed)

        return target_latents.retrieve()
