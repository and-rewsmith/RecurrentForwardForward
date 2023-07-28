import logging

import torch

from RecurrentFF.model.data_scenario.processor import DataScenarioProcessor
from RecurrentFF.util import layer_activations_to_goodness, ForwardMode
from RecurrentFF.settings import Settings


class StaticSingleClassProcessor(DataScenarioProcessor):
    def __init__(self, inner_layers, data_config):
        self.inner_layers = inner_layers
        self.data_config = data_config
        self.settings = Settings()

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
            if limit_batches != None and batch == limit_batches:
                break

            logging.info("Starting inference for test batch: " + str(batch))

            # tuple: (correct, total)
            accuracy_contexts = []

            with torch.no_grad():
                data, labels = test_data
                data = data.to(self.settings.device.device)
                labels = labels.to(self.settings.device.device)

                iterations = data.shape[0]

                all_labels_goodness = []

                # evaluate goodness for each possible label
                for label in range(self.data_config.num_classes):
                    self.inner_layers.reset_activations(False)

                    one_hot_labels = torch.zeros(
                        data.shape[1], self.data_config.num_classes, device=self.settings.device.device)
                    one_hot_labels[:, label] = 1.0

                    for _preinit_iteration in range(0, len(self.inner_layers)):
                        self.inner_layers.advance_layers_forward(ForwardMode.PredictData,
                                                                 data[0], one_hot_labels, False)

                    lower_iteration_threshold = iterations // 2 - \
                        self.data_config.focus_iteration_neg_offset
                    upper_iteration_threshold = iterations // 2 + \
                        self.data_config.focus_iteration_pos_offset
                    goodnesses = []
                    for iteration in range(0, iterations):
                        self.inner_layers.advance_layers_forward(ForwardMode.PredictData,
                                                                 data[iteration], one_hot_labels, True)

                        if iteration >= lower_iteration_threshold and iteration <= upper_iteration_threshold:
                            layer_goodnesses = []
                            for layer in self.inner_layers:
                                layer_goodnesses.append(layer_activations_to_goodness(
                                    layer.predict_activations.current))

                            goodnesses.append(torch.stack(
                                layer_goodnesses, dim=1))

                    # tensor of shape (batch_size, iterations, num_layers)
                    goodnesses = torch.stack(goodnesses, dim=1)
                    # average over iterations
                    goodnesses = goodnesses.mean(dim=1)
                    # average over layers
                    goodness = goodnesses.mean(dim=1)

                    logging.debug("goodness for prediction" + " " +
                                  str(label) + ": " + str(goodness))
                    all_labels_goodness.append(goodness)

                all_labels_goodness = torch.stack(all_labels_goodness, dim=1)

                # select the label with the maximum goodness
                predicted_labels = torch.argmax(all_labels_goodness, dim=1)
                logging.debug("predicted labels: " + str(predicted_labels))
                logging.debug("actual labels: " + str(labels))

                total = data.size(1)
                correct = (predicted_labels == labels).sum().item()

                accuracy_contexts.append((correct, total))

        total_correct = sum(correct for correct, _total in accuracy_contexts)
        total_submissions = sum(
            total for _correct, total in accuracy_contexts)
        accuracy = total_correct / total_submissions * 100 if total_submissions else 0
        logging.info(f'test accuracy: {accuracy}%')

        return accuracy
