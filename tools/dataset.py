import torch
import torchvision.transforms as TF
from torch.utils.data import Dataset

import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from tools.dataset_utils import box_to_label
from tools.visualization import DetectionVisualization
from tools.parse_raw_dataset import class_names


# get full path to the root directory of this project
proj_root = os.path.dirname(os.path.abspath(__file__)).split(os.sep)
proj_root = os.path.join("\\", *proj_root[:-1])
proj_root = proj_root[:2] + '\\' + proj_root[2:]
data_dir = os.path.join(proj_root, 'data')


class VOC(Dataset):
    """Dataset object representing VOC dataset"""
    def __init__(self, split, transform=None, image_size=(384, 384), len_tiny=500,
                 model_downsample_factor=4.0,
                 nClass=20):
        """Constructor of class VOC

        Args:
            split (str): split of dataset (train, test, tiny)
            transform (torchvision.transforms): transformation to apply on images
            image_size (tuple[int]): size of image fed into model
            len_tiny (int): len of tiny set
            model_downsample_factor (float): model's downsample factor
            nClass (int): number of classes in VOC
        """

        if not os.path.exists(os.path.join(data_dir, "train_images.json")) or not \
                os.path.exists(os.path.join(data_dir, "test_images.json")):
            print("Raw dataset has not been parsed. Execute tools/parse_raw_dataset.py first. Exitting")
            exit(1)

        assert split.lower() in ('train', 'test', 'tiny')
        self.split = split.lower()
        self.downsample_factor = model_downsample_factor
        self.output_size = (int(float(image_size[0]) / self.downsample_factor),
                            int(float(image_size[1]) / self.downsample_factor))
        self.nClass = 20
        if transform is None:
            self.transform = TF.Compose([
                TF.Resize(image_size),
                TF.ToTensor(),
                TF.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = transform
            print("Using customized transform")
            print("[REMINDER] remember to normalize image according to ImagesNet stats")

        _split = 'test' if self.split == 'test' else 'train'
        with open(os.path.join(data_dir, "{}_images.json".format(_split)), 'r') as f:
            self.images_path = json.load(f)
            # test number of images
            assert len(self.images_path) == 4952 if _split == 'test' else 16551, \
                "Something went wrong, recreate {}_images.json file to be sure".format(_split)
            if self.split == 'tiny':
                self.images_path = self.images_path[:len_tiny]


        with open(os.path.join(data_dir, "{}_annotations.json".format(_split)), 'r') as f:
            self.anns = json.load(f)
            if self.split == 'tiny':
                self.anns = self.anns[:len_tiny]

        nObjects = 0
        for ann in self.anns:
            nObjects += len(ann['boxes'])

        # test number of objects
        if self.split != 'tiny':
            assert nObjects == 14976 if _split == 'test' else 47223, \
                "Something went wrong, recreate {}_annotations.json file to be sure".format(_split)

        print("%s set has %d images with the total of %d boxes" % (self.split, len(self.images_path), nObjects))

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item_idx):
        # create model's input
        image = Image.open(self.images_path[item_idx])
        image = self.transform(image)  # a torch.Tensor of shape (3, 384, 384)

        # extract image's annotation from self.anns
        boxes = self.anns[item_idx]['boxes']  # list (nBoxes), each box is [xmin, ymin, xmax, ymax]
        labels = self.anns[item_idx]['labels']  # list (nBoxes), each label is int -
        # Hint: keys of a dict stored in self.anns is defined in function `parse_annotation` in parse_raw_dataset.py

        # initialize heat_map, offset_map, size_map as zeros tensor with appropriate size using np.zeros
        heat_map = np.zeros((self.nClass, self.output_size[0], self.output_size[1]))
        offset_map = np.zeros((2, self.output_size[0], self.output_size[1]))  #
        size_map = np.zeros((2, self.output_size[0], self.output_size[1]))  #

        # fill those 3 maps above using elements in `boxes` and `labels`
        for box, obj_type in zip(boxes, labels):
            box_to_label(box, obj_type, heat_map, offset_map, size_map) #

        # convert 3 maps above into torch.Tensor and concatenate them along the channel dimension
        net_label = torch.cat([torch.tensor(heat_map, dtype=torch.float),
                               torch.tensor(offset_map, dtype=float),
                               torch.tensor(size_map, dtype=torch.float)], dim=0)  # shape (nClasses+4, 96, 96)

        return image, net_label


if __name__ == '__main__':
    # This part just to display some samples taken from an instance of class VOC
    dataset = VOC('tiny')
    visualizer = DetectionVisualization()
    tf = TF.Compose([TF.Resize((96, 96)), TF.ToPILImage()])

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    o = 10
    for idx in range(25):
        image, net_label= dataset[idx + o]
        labels = dataset.anns[idx]['labels']
        labels = list(set(labels))

        # unnormalize image
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        for i in range(3):
            image[i] = image[i] * std[i] + mean[i]
        # resize image & convert it from torch.Tensor to PIL.Image
        image = tf(image)

        # get heatmap
        heatmap = net_label[:20].unsqueeze(0)

        # draw
        r, c = idx // 5, idx % 5
        label_idx = 0 if len(labels) == 1 else 1
        visualizer.draw_one_image(image, axes[r, c])
        visualizer.draw_heatmap(heatmap, axes[r, c], [labels[label_idx]], 0.55)
        axes[r, c].set_xticks([])
        axes[r, c].set_yticks([])
        axes[r, c].set_title("class {}".format(class_names[labels[label_idx]]), fontdict={'fontsize': 13})

    fig.tight_layout()
    plt.show()

