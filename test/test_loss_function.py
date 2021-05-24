import unittest
import torch
import numpy as np

from model.loss_function import CenterNetLoss


class TestLossFunction(unittest.TestCase):
    def test_focal_loss(self):
        loss_func = CenterNetLoss()
        input = torch.zeros(1, 3, 2, 3) + 0.5
        target = torch.zeros_like(input)
        target[0, 0, 1, 0] = 1.0
        target[0, 1, 0, 1] = 1.0
        target[0, 2, 1, 1] = 1.0
        mask_true_center = target > 0.999
        focal_loss = loss_func.focal_loss(input, target, mask_true_center)
        expected_focal_loss = -1.5 * np.log(0.5)
        self.assertTrue(np.abs(expected_focal_loss - focal_loss.item()) < 1e-5)

        # perfect prediction
        focal_loss = loss_func.focal_loss(target, target, mask_true_center)
        self.assertTrue(np.abs(focal_loss.item()) < 1e-5)

    def test_forward(self):
        loss_func = CenterNetLoss(debug=True, nClass=3)
        input = torch.zeros(1, 7, 2, 3, dtype=torch.float) + 0.5
        target = torch.zeros_like(input, dtype=torch.float)
        target[0, 0, 1, 0] = 1.0
        target[0, 1, 0, 1] = 1.0
        target[0, 2, 1, 1] = 1.0
        target[0, 3] += 2.0
        target[0, 4] += 3.0
        target[0, 5] += 0.1
        target[0, 6] += 0.2
        losses = loss_func(input, target)  # focal, size, offset
        expected_losses = (-1.5 * np.log(0.5), 4.0, 0.7)  # focal, size, offset
        for loss, expected_loss in zip(losses, expected_losses):
            self.assertTrue(np.abs(loss - expected_loss) < 1e-5)


if __name__ == '__main__':
    unittest.main()
