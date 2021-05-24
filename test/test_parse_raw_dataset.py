import unittest
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from tools.parse_raw_dataset import *


colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080']  # 20 distinct colors (source: google search)


def draw_annotation(axe, anns):
    for box, label in zip(anns['boxes'], anns['labels']):
        xmin, ymin, xmax, ymax = box
        obj_type = class_names[label]
        corners = np.array([
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax]
        ])
        corner_draw_order = list(range(-1, 4))
        axe.plot(corners[corner_draw_order, 0], corners[corner_draw_order, 1], linewidth=2, color=colors[label])
        # put a name to it
        text_x, text_y = corners[0, 0] + 5, corners[0, 1] -5
        axe.text(text_x, text_y, obj_type,
                 bbox=dict(boxstyle="square", ec=colors[label], fc=colors[label]))


# get full path to the root directory of this project
#proj_root = os.path.dirname(os.path.abspath(__file__)).split('/')
proj_root = r'C:\Users\alika\Desktop\2.semestar\AIRO\Lab1\''


class TestParseRawDataset(unittest.TestCase):
    def test_create_data_lists(self):
        # test number of images in test set
        #with open(os.path.join(proj_root, 'data', 'test_images.json'), 'r') as f:
        with open(r'C:/Users/alika/Desktop/2.semestar/AIRO/Lab1/data/test_images.json', 'r') as f:
            test_images = json.load(f)
        self.assertEqual(len(test_images), 4952, msg="Expect 4952 images in test set, get {}".format(
            len(test_images)
        ))

        # test number of objects in test set
        #with open(os.path.join(proj_root, 'data', 'test_annotations.json'), 'r') as f:
        with open(r'C:/Users/alika/Desktop/2.semestar/AIRO/Lab1/data/test_annotations.json', 'r') as f:
            test_annotations = json.load(f)
        nObjects = 0
        for ann in test_annotations:
            nObjects += len(ann['boxes'])
        self.assertEqual(nObjects, 14976, msg="Expect 14976 objects in test set, get {}".format(nObjects))

        # display annotation
        # fig, axe = plt.subplots(1, 3)
        # for i in range(3):
        #     image_path = r"{}".format(test_images[i])
        #     image = Image.open(image_path)
        #     image.resize((384, 384))
        #     annotation = test_annotations[i]
        #     axe[i].imshow(image)
        #     draw_annotation(axe[i], annotation)
        # plt.show()


if __name__ == '__main__':
    unittest.main()
