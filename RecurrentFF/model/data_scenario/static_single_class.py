import logging
from typing import List, Optional, cast, Tuple
from pyparsing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import RMSprop, Optimizer
import wandb

from RecurrentFF.model.data_scenario.processor import DataScenarioProcessor
from RecurrentFF.model.inner_layers import InnerLayers
from RecurrentFF.util import Activations, LatentAverager, TrainLabelData, layer_activations_to_badness, ForwardMode, is_confident, zero_correct_class_softmax, swap_top_two_softmax, zero_highest_logit, sample_from_logits_excluding_highest, standardize_layer_activations
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
    def __init__(self, inner_layers: InnerLayers, settings: Settings):
        self.settings = settings
        self.inner_layers = inner_layers

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
            generative_linear: torch.nn.modules.container.Sequential,
            optimizer: torch.optim.Optimizer,
            limit_batches: Optional[int] = None,
            is_test_set: bool = False,
            write_activations: bool = False,
    ) -> float:
        if write_activations:
            assert self.settings.data_config.test_batch_size == 1 \
                and is_test_set, "Cannot write activations for batch size > 1"
            activity_tracker = StaticSingleClassActivityTracker()

        forward_mode = ForwardMode.PredictData if is_test_set else ForwardMode.PositiveData

        accuracy_contexts = []

        for batch, test_data in enumerate(loader):
            if limit_batches is not None and batch == limit_batches:
                break

            data, labels = test_data
            data = data.to(self.settings.device.device)
            labels = labels.to(self.settings.device.device)

            if write_activations:
                activity_tracker.reinitialize(data, labels)

            # since this is static singleclass we can use the first frame
            # for the label
            labels = labels[0]

            iterations = data.shape[0]

            all_labels_badness = []

            self.inner_layers.reset_activations(True)

            upper_clamped_tensor = self.get_preinit_upper_clamped_tensor(
                (data.shape[1], self.settings.data_config.num_classes))

            for _preinit_step in range(
                    0, self.settings.model.prelabel_timesteps):
                self.inner_layers.advance_layers_forward(
                    ForwardMode.PositiveData, data[0], upper_clamped_tensor, False)
                self.inner_layers.advance_layers_forward(
                    ForwardMode.NegativeData, data[0], upper_clamped_tensor, False)

                if write_activations:
                    activity_tracker.track_partial_activations(
                        self.inner_layers)

            lower_iteration_threshold = iterations // 2 - \
                iterations // 10
            upper_iteration_threshold = iterations // 2 + \
                iterations // 10
            badnesses = []
            class_predictions_agg = torch.zeros(
                data.shape[1], self.settings.data_config.num_classes).to(self.settings.device.device)
            for iteration in range(0, iterations // 3 * 2):
                # for iteration in range(0, 1):
                iteration = min(iteration, iterations - 1)

                generative_output = generative_linear(
                    torch.cat([layer.pos_activations.current for layer in self.inner_layers], dim=1))
                assert generative_output.shape[0] == data.shape[1] and generative_output.shape[
                    1] == self.settings.data_config.data_size + self.settings.data_config.num_classes
                reconstructed_data, reconstructed_labels = generative_output.split(
                    [self.settings.data_config.data_size, self.settings.data_config.num_classes], dim=1)
                assert reconstructed_data.shape[1] == self.settings.data_config.data_size

                reconstructed_labels_softmax = F.softmax(
                    reconstructed_labels, dim=1)
                assert reconstructed_labels.shape[1] == self.settings.data_config.num_classes
                assert reconstructed_labels.shape[0] == data.shape[1]
                reconstructed_labels = torch.argmax(
                    reconstructed_labels_softmax, dim=1)
                assert reconstructed_labels.shape == labels.shape
                correct = (reconstructed_labels == labels).sum().item()
                # print(correct / data.shape[1] * 100)
                total = data.size(1)
                generative_output = generative_output.detach()

                input_data_sample = (
                    data[iteration],
                    generative_output[:, 0:self.settings.data_config.data_size])
                    # data[iteration])
                label_data_sample = (
                    torch.softmax(
                        generative_output[:, self.settings.data_config.data_size:], dim=1),
                    # torch.softmax(generative_output[:, self.settings.data_config.data_size:], dim=1),
                    # torch.zeros(data.size(1), self.settings.data_config.num_classes).to(
                    #     self.settings.device.device),
                    # torch.zeros(data.size(1), self.settings.data_config.num_classes).to(
                    #     self.settings.device.device),
                    # torch.softmax(
                    #     generative_output[:, self.settings.data_config.data_size:], dim=1),
                    # torch.softmax(
                    #     generative_output[:, self.settings.data_config.data_size:], dim=1),
                    # zero_highest_logit(
                    #     generative_output[:, self.settings.data_config.data_size:])
                    # torch.softmax(generative_output[:, self.settings.data_config.data_size:], dim=1),
                    # sample_from_logits(zero_highest_logit(
                    #     generative_output[:, self.settings.data_config.data_size:])),
                    # sample_from_logits_excluding_highest(generative_output[:, self.settings.data_config.data_size:]),
                    # torch.nn.functional.one_hot(torch.argmax(
                    #     generative_output[:, self.settings.data_config.data_size:], dim=1), num_classes=10).to(dtype=torch.float32, device=self.settings.device.device),
                    swap_top_two_softmax(torch.softmax(
                        generative_output[:, self.settings.data_config.data_size:], dim=1))
                )
                self.inner_layers.advance_layers_train(
                    input_data_sample, label_data_sample, True, None)

                pre_op_grad = self.inner_layers.layers[0].forward_linear.weight.grad[0][2].item(
                )
                pre_opt_softmax_predicted_classes = torch.softmax(
                    generative_output[:, self.settings.data_config.data_size:], dim=1)
                self.optimizer.zero_grad()
                post_opt_logits = generative_linear(
                    torch.cat([layer.pos_activations.current for layer in self.inner_layers], dim=1))[:, self.settings.data_config.data_size:]
                post_op_log_softmax_predicted_classes = torch.log_softmax(
                    post_opt_logits, dim=1)
                criterion = torch.nn.KLDivLoss()
                loss = criterion(
                    post_op_log_softmax_predicted_classes, pre_opt_softmax_predicted_classes)
                loss.backward()

                # from torchviz import make_dot
                # from itertools import chain
                # # Generate the dot graph
                # def combine_iterators_with_prefix(*iterators: Iterator[Tuple[str, nn.Parameter]]) -> dict:
                #     result = {}
                #     for i, iterator in enumerate(iterators):
                #         for name, param in iterator:
                #             result[f"{i}_{name}"] = param
                #     return result
                # params = combine_iterators_with_prefix(generative_linear.named_parameters(), self.inner_layers.layers[0].filtered_named_parameters(), self.inner_layers.layers[1].filtered_named_parameters(), self.inner_layers.layers[2].filtered_named_parameters(), self.inner_layers.layers[3].filtered_named_parameters(), self.inner_layers.layers[4].filtered_named_parameters(), self.inner_layers.layers[5].filtered_named_parameters())
                # print("Params keys:")
                # for key in list(params.keys())[:200]:  # Print first 20 keys
                #     print(f"  {key}")
                # input()
                # dot = make_dot(loss, params=params)
                # # Save the graph to a file (you can use .png, .svg, .pdf, etc.)
                # dot.render("computation_graph", format="png", cleanup=True)
                # input("resume")

                post_op_grad = self.inner_layers.layers[0].forward_linear.weight.grad[0][2].item(
                )
                # assert pre_op_grad != post_op_grad

                # optimizer.step()
                # for layer in self.inner_layers:
                #     layer.optimizer.step()
                for layer in self.inner_layers:
                    layer.pos_activations.current = layer.pos_activations.current.clone().detach()
                    layer.neg_activations.current = layer.neg_activations.current.clone().detach()
                    layer.pos_activations.previous = layer.pos_activations.previous.clone().detach()
                    layer.neg_activations.previous = layer.neg_activations.previous.clone().detach()

                class_predictions_agg += torch.softmax(
                    generative_output[:, self.settings.data_config.data_size:], dim=1)

                if write_activations:
                    activity_tracker.track_partial_activations(
                        self.inner_layers)

                # conf, should_stop = is_confident(reconstructed_labels_softmax, torch.nn.functional.one_hot(labels, 10), .5)
                # if should_stop and iteration > 3:
                #     print(iteration)
                #     break

            correct_number_agg = (torch.argmax(
                class_predictions_agg, dim=1) == labels).float().sum().item()
            accuracy_contexts.append((correct_number_agg, data.size(1)))

        # ###
        # import matplotlib.pyplot as plt
        # # Calculate sum of squared activations and separate correct/incorrect predictions
        # predictions = torch.argmax(class_predictions_agg, dim=1)
        # correct_mask = predictions == labels
        # top_quartile_stats = []

        # for layer in self.inner_layers:
        #     activations = layer.pos_activations.current
        #     squared_sums = torch.sum(activations ** 2, dim=1)

        #     # Calculate top quartile statistics
        #     top_quartile_threshold = torch.quantile(squared_sums, 0.8)
        #     top_quartile_mask = squared_sums >= top_quartile_threshold
        #     top_quartile_correct = torch.sum(correct_mask & top_quartile_mask).item()
        #     top_quartile_incorrect = torch.sum(~correct_mask & top_quartile_mask).item()
        #     top_quartile_stats.append((top_quartile_correct, top_quartile_incorrect))

        # # Plot horizontal bar chart for quartiles analysis
        # num_layers = len(self.inner_layers)
        # fig, ax = plt.subplots(figsize=(12, num_layers * 0.5 + 2))  # Adjust figure height based on number of layers

        # layer_names = [f'Layer {i+1}' for i in range(num_layers)]
        # correct_counts = [stats[0] for stats in top_quartile_stats]
        # incorrect_counts = [stats[1] for stats in top_quartile_stats]

        # ax.barh(layer_names, correct_counts, label='Correct', color='green', alpha=0.7)
        # ax.barh(layer_names, incorrect_counts, left=correct_counts, label='Incorrect', color='red', alpha=0.7)

        # ax.set_title('Top Quartile of Activations by Layer')
        # ax.set_xlabel('Count')
        # ax.set_ylabel('Layers')
        # ax.legend(loc='lower right')

        # # Add text labels
        # for i, (correct, incorrect) in enumerate(zip(correct_counts, incorrect_counts)):
        #     total = correct + incorrect
        #     ax.text(total/2, i, f'{correct}/{total}',
        #             ha='center', va='center', color='black', fontweight='bold')

        # plt.tight_layout()
        # plt.savefig(f'quartile_analysis_batch_{batch}.png')
        # plt.close()
        # ###

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
