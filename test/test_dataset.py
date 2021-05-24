import unittest
import torch

from tools.dataset import VOC


class TestDataset(unittest.TestCase):
    def test_dataset_len(self):
        len_tiny = 100
        tiny = VOC('tiny', len_tiny=len_tiny)
        self.assertEqual(len_tiny, len(tiny))

    def test_tensor_size(self):
        tiny = VOC('tiny', len_tiny=1)
        image, net_label = tiny[0]
        self.assertTrue(isinstance(image, torch.Tensor), msg="image must be a torch.Tensor")
        self.assertTrue(isinstance(net_label, torch.Tensor), msg="net_label must be a torch.Tensor")
        self.assertTrue(torch.all(torch.tensor(image.size()) == torch.tensor([3, 384, 384])).item(),
                        msg="Image has wrong dimension, expect {}, got {}".format(
                            torch.tensor([3, 384, 384]), image.size()
                        ))
        self.assertTrue(torch.all(torch.tensor(net_label.size()) == torch.tensor([24, 96, 96])).item(),
                        msg="Image has wrong dimension, expect {}, got {}".format(
                            torch.tensor([24, 96, 96]), net_label.size()
                        ))



if __name__ == '__main__':
    unittest.main()
