import unittest
from tools.dataset_utils import *


class TestDatasetUtils(unittest.TestCase):
    def test_compute_iou(self):
        box1 = [0, 0, 1.0, 1.0]  # [xmin, ymin, xmax, ymax]
        iou = compute_iou(box1, box1)
        self.assertEqual(1.0, iou)

        box2 = [2.0, 2.0, 3.0, 3.0]
        iou = compute_iou(box1, box2)
        self.assertEqual(0.0, iou)

        box1 = [0., 0., 2., 2.]
        box2 = [1.0, 0.0, 3.0, 3.0]
        iou = compute_iou(box1, box2)
        self.assertEqual(0.25, iou)

    def test_box_compute_cornerNet_radius(self):
        box1 = [0, 0, 2, 2]  # [xmin, ymin, xmax, ymax]
        h, w = box1[3] - box1[1], box1[2] - box1[0]
        min_iou = 0.3
        r = box_compute_cornerNet_radius(box1, min_iou)
        theta = np.arctan2(h, w)
        box2 = [-r * np.cos(theta), -r * np.sin(theta), w + r * np.cos(theta), h + r * np.sin(theta)]
        iou = compute_iou(box1, box2)
        self.assertGreater(iou, min_iou)

    def test_draw_gaussian_on_matrix(self):
        draw_top_left = np.zeros((10, 10))
        draw_gaussian_on_matrix([0, 0, 1, 1], draw_top_left, sigma=1.0, center=[0.0, 0.0], model_downsample_factor=1.0)
        expected_top_left = np.zeros_like(draw_top_left)
        for r in range(4):
            for c in range(4):
                expected_top_left[r, c] = np.exp(-(c**2 + r**2)/2)
        self.assertTrue(np.all(expected_top_left == draw_top_left),
                        msg="Fail when center at top left\n Expected:\n {}\n---\n Get:\n {}".format(
                            expected_top_left, draw_top_left
                        ))

        draw_bottom_right = np.zeros((10, 10))
        draw_gaussian_on_matrix([0, 0, 1, 1], draw_bottom_right, sigma=1.0, center=[9.0, 9.0],
                                model_downsample_factor=1.0)
        expected_bottom_right = np.zeros_like(draw_bottom_right)
        for r in range(6, 10):
            for c in range(6, 10):
                expected_bottom_right[r, c] = np.exp(-((c - 9.0) ** 2 + (r - 9.0) ** 2) / 2)
        self.assertTrue(np.all(expected_bottom_right == draw_bottom_right),
                        msg="Fail when center at top left\n Expected:\n {}\n---\n Get:\n {}".format(
                            expected_bottom_right, draw_bottom_right
                        ))

        draw_middle = np.zeros((10, 10))
        draw_gaussian_on_matrix([0, 0, 1, 1], draw_middle, sigma=1.0, center=[4.0, 4.0], model_downsample_factor=1.0)
        expected_middle = np.zeros_like(draw_middle)
        for r in range(1, 8):
            for c in range(1, 8):
                expected_middle[r, c] = np.exp(-((c - 4.0) ** 2 + (r - 4.0) ** 2) / 2)
        self.assertTrue(np.all(expected_middle == draw_middle),
                        msg="Fail when center at top left\n Expected:\n {}\n---\n Get:\n {}".format(
                            expected_middle, draw_middle
                        ))

    def test_box_to_label(self):
        box = [14, 14, 19, 19]
        obj_type = 10
        heat_map = np.zeros((20, 10, 10))
        offset_map = np.zeros((2, 10, 10))
        size_map = np.zeros((2, 10, 10))
        box_to_label(box, obj_type, heat_map, offset_map, size_map)

        # check heat_map
        for i in range(20):
            target_prob = 0.0 if i != obj_type else 1.0
            self.assertEqual(target_prob, heat_map[i, 4, 4],
                             msg="Gaussian is placed at the wrong channel or wrong location. "
                                 "Peak must be at (4, 4) of channel 10")
        # check offset_map
        for i in range(2):
            offset_type = 'x' if i == 0 else 'y'
            self.assertEqual(0.125, offset_map[i, 4, 4], msg="Wrong offset for {}".format(offset_type))
        offset_map[:, 4, 4] = np.zeros(2)  # to make the following test convenient by using np.all
        self.assertTrue(np.all(offset_map == 0), msg="Only pixel at low-resolution center has non-zero offset value")

        # check size_map
        for i in range(2):
            size_type = 'w' if i == 0 else 'h'
            self.assertEqual(1.25, size_map[i, 4, 4], msg="Wrong offset for {}".format(size_type))
        size_map[:, 4, 4] = np.zeros(2)  # to make the following test convenient by using np.all
        self.assertTrue(np.all(size_map == 0), msg="Only pixel at low-resolution center has non-zero size value")


if __name__ == '__main__':
    unittest.main()
