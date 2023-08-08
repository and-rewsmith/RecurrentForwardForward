import unittest

import torch

from RecurrentFF.model.data_scenario.static_single_class import (
    formulate_incorrect_class
)
from RecurrentFF.settings import Settings


TEST_CONFIG_FILE = "./test/config-files/smoke.toml"


class TestFormulateIncorrectClass(unittest.TestCase):
    def test_shape(self):
        mock_settings = Settings.from_config_file(
            TEST_CONFIG_FILE)

        prob_tensor = torch.tensor([[0.1, 0.3, 0.6], [0.5, 0.2, 0.3]])
        correct_onehot_tensor = torch.tensor([[0, 0, 1], [1, 0, 0]])

        result = formulate_incorrect_class(
            prob_tensor, correct_onehot_tensor, mock_settings)
        self.assertEqual(result.shape, prob_tensor.shape,
                         "The result shape should match the input shape.")

    def test_prob_all_zero(self):
        mock_settings = Settings.from_config_file(
            TEST_CONFIG_FILE)

        prob_tensor = torch.zeros(3, 3)
        correct_onehot_tensor = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        result = formulate_incorrect_class(
            prob_tensor, correct_onehot_tensor, mock_settings)

        # Check that the result is not the same as the correct_onehot_tensor
        self.assertFalse(
            torch.equal(
                result,
                correct_onehot_tensor),
            "The result should not match the correct classes when all probabilities are zero.")

    def test_masking_correct_class(self):
        mock_settings = Settings.from_config_file(
            TEST_CONFIG_FILE)

        prob_tensor = torch.tensor([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3]])
        correct_onehot_tensor = torch.tensor([[0, 0, 1], [0, 1, 0]])

        result = formulate_incorrect_class(
            prob_tensor, correct_onehot_tensor, mock_settings)
        correct_indices = correct_onehot_tensor.argmax(dim=1)
        selected_indices = result.argmax(dim=1)

        # Check that no selected index matches the correct index
        self.assertFalse(torch.any(selected_indices == correct_indices),
                         "No selected index should match the correct index.")

    def test_high_confidence_good_classifier(self):
        mock_settings = Settings.from_config_file(
            TEST_CONFIG_FILE)

        prob_tensor = torch.tensor([[1.0, 0, 0], [0, 1.0, 0]])
        correct_onehot_tensor = torch.tensor([[1, 0, 0], [0, 1, 0]])

        result = formulate_incorrect_class(
            prob_tensor, correct_onehot_tensor, mock_settings)
        correct_indices = correct_onehot_tensor.argmax(dim=1)
        selected_indices = result.argmax(dim=1)
        self.assertFalse(torch.any(selected_indices == correct_indices),
                         "No selected index should match the correct index.")


if __name__ == '__main__':
    unittest.main()
