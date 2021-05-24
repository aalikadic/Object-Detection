import unittest
import torch

from model.model_utils import *


class TestModelUtils(unittest.TestCase):
    def test_idx1d_to_indices3d(self):
        idx_1d = torch.tensor([3, 7, 16])  # index in the flatten version of
        # a 3d tensor having the size (3, 2, 3)

        expected_result = (
            torch.tensor([0, 1, 2]),  # channel index
            torch.tensor([1, 0, 1]),  # row index
            torch.tensor([0, 1, 1])  # col index
        )
        ch, row, col = idx1d_to_indices3d(idx_1d, nRows=2, nCols=3)
        self.assertTrue(torch.all(ch == expected_result[0]).item(), msg="Wrong channel index")
        self.assertTrue(torch.all(row == expected_result[1]).item(), msg="Wrong row index")
        self.assertTrue(torch.all(col == expected_result[2]).item(), msg="Wrong column index")

    def test_nms_heat_map(self):
        heatmap = torch.rand(5, 5)
        heatmap[[1, 1, 3, 3], [1, 3, 3, 1]] = torch.tensor([2, 3, 4, 5], dtype=torch.float)

        expected_result = torch.zeros_like(heatmap, dtype=torch.float)
        expected_result[[1, 1, 3, 3], [1, 3, 3, 1]] = torch.tensor([2, 3, 4, 5], dtype=torch.float)

        # convert heatmap to 4D tensor of shape (1, 1, 5, 5)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)

        nms_heatmap = nms_heat_map(heatmap)
        self.assertTrue(torch.all(nms_heatmap.squeeze() == expected_result).item(),
                        msg="Wrong NMS.\n Input = \n{}\n---\n Output=\n{}\n---\n Expected Output = \n{}".format(
                            heatmap.squeeze(), nms_heatmap.squeeze(), expected_result
                        ))

    def test_get_topK_peaks(self):
        heatmap = torch.rand(5, 5)
        heatmap[[1, 1, 3, 3], [1, 3, 3, 1]] = torch.tensor([2, 3, 4, 5], dtype=torch.float)

        expected_result = (
            torch.tensor([0]*4, dtype=torch.long),  # channel index, Note: [0] * 4 == [0, 0, 0, 0]
            torch.tensor([3, 3, 1, 1], dtype=torch.long),  # row index (i.e. ys)
            torch.tensor([1, 3, 3, 1], dtype=torch.long),  # column index (i.e. xs)
            torch.tensor([5, 4, 3, 2], dtype=torch.float)  # class prob (i.e. score)
        )

        chs, ys, xs, score = get_topK_peaks(heatmap.unsqueeze(0), topK=4)
        self.assertTrue(torch.all(chs == expected_result[0]).item(),
                        msg="Wrong channel index.\n Input = \n{}\n---\n Output = \n{}\n---\n ExpectedOut = \n{}".format(
                            heatmap, chs, expected_result[0]
                        ))
        self.assertTrue(torch.all(ys == expected_result[1]).item(),
                        msg="Wrong row index.\n Input = \n{}\n---\n Output = \n{}\n---\n ExpectedOut = \n{}".format(
                            heatmap, ys, expected_result[1]
                        ))
        self.assertTrue(torch.all(xs == expected_result[2]).item(),
                        msg="Wrong column index.\n Input = \n{}\n---\n Output = \n{}\n---\n ExpectedOut = \n{}".format(
                            heatmap, xs, expected_result[2]
                        ))
        self.assertTrue(torch.all(score == expected_result[3]).item(),
                        msg="Wrong class prob.\n Input = \n{}\n---\n Output = \n{}\n---\n ExpectedOut = \n{}".format(
                            heatmap, score, expected_result[3]
                        ))


if __name__ == '__main__':
    unittest.main()
