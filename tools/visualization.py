import torch
from PIL import Image
import numpy as np
from matplotlib.axes import Axes
from typing import Union


class DetectionVisualization:
	"""Provide functionalities for plotting detection"""
	def __init__(self, vis_size=(96, 96), class_names=None, text_offset=(1, -2), score_threshold=0.3):
		"""Constructor

		Args:
			image_dir (str): full path to dataset's image folder
			vis_size (tuple[int]): size of image for visualization (equal to size of model's output)
			class_names (list[str]): list of classes of objects in the dataset
			text_offset (tuple[float]): offset (in pixel) of box's name w.r.t box's top-left corner
			score_threshold (float): threshold for class probability
		"""
		self.class_names = class_names if class_names else \
			["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
			 "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
			 "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

		self.class_to_int = dict(zip(self.class_names, range(len(self.class_names))))
		# this is to convert a class in the form of string into an integer

		self.colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
          '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
          '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
          '#000075', '#808080']  # 20 distinct colors (source: google search)

		self.vis_size = vis_size
		self.text_offset = text_offset
		self.score_threshold = score_threshold

	def draw_one_image(self, image, axe):
		"""Draw 1 image

		Args:
			image (Union[str, Image]): image to draw
			axe (Axes): axe to draw on
		"""
		if isinstance(image, str):
			# image is currently a path to an image
			image = Image.open(image)
		# resize the image to the visualization size
		if image.height != self.vis_size[0] or image.width != self.vis_size[1]:
			image = image.resize(self.vis_size, Image.ANTIALIAS)
		axe.imshow(image)

	def draw_images(self, images, axes):
		"""Draw a list of images

		Args:
			images (list[Union[str, Image]]): list of image names or images
			axes (Union[np.ndarray,  Axes]): axes to draw on
		"""
		assert isinstance(images, list), "images must be of a list (currently {})".format(type(images))
		if len(images) == 1:
			assert not isinstance(axes, np.ndarray), \
				"There is just 1 image, no need a grid ({}) of Axes".format(axes.shape)
			self.draw_one_image(images[0], axes)
		else:
			assert len(images) == axes.size, \
				"Need {} Axes to draw (currently {})".format(len(images), axes.size)
			nCols = axes.shape[1]
			for i, image in enumerate(images):
				row_idx, col_idx = i // nCols, i % nCols
				self.draw_one_image(image, axes[row_idx, col_idx])

	def draw_heatmap(self, heatmap, axes, gt_obj, alpha=0.7):
		"""Draw heatmap overlay on images

		Args:
			heatmap (torch.Tensor): a single heat map, shape (N, nClasses, H, W)
			axes (Union[np.ndarray, Axes]): axe on which heatmap & image are drew
			gt_obj (list[Union[str, int]]): list of chosen class object to be displayed for each image in the batch,
				one class of obj/image. In the case of images have multiple classes, only the chosen one has
				its heatmap displayed
			alpha (float): blending coefficient, in [0.0, 1.0], the higher the clearer the heatmap
		"""
		assert isinstance(gt_obj, list), "gt_obj must be a list (currently {})".format(type(gt_obj))
		assert len(gt_obj) == heatmap.size(0), "Must choose 1 class obj to display heatmap for each image"
		if heatmap.size(0) > 1:
			assert isinstance(axes, np.ndarray), "Batch has >1 images, need a grid of Axes"
			assert heatmap.size(0) == axes.size, \
				"Need {} Axes to draw (currently {})".format(heatmap.size(0), axes.size)
		else:
			assert not isinstance(axes, np.ndarray), "Batch has only 1 images, don't need a grid of Axes"

		if isinstance(gt_obj[0], str):
			# get the int representing gt_obj
			gt_obj = [self.class_to_int[o] for o in gt_obj]

		if heatmap.size(0) == 1:
			# draw only 1 heatmap
			chosen_map = heatmap[0, gt_obj[0]]  # shape (H, W)
			axes.imshow(chosen_map, alpha=alpha)
		else:
			nCols = axes.shape[1]
			for i in range(heatmap.size(0)):
				row_idx, col_idx = i // nCols, i % nCols
				chosen_map = heatmap[i, gt_obj[i]]  # shape (H, W)
				axes[row_idx, col_idx].imshow(chosen_map, alpha=alpha)

	def draw_box(self, box, axe):
		"""Draw a box on an Axe

		Args:
			box (Union[numpy.ndarray, torch.Tensor]): shape (6), (cx, cy, w, h, class_int, score)
			axe (Axes): axe to draw on
		"""
		if isinstance(box, torch.Tensor):
			box = box.numpy()
		cx, cy, w, h, obj_type, score = box
		obj_type = int(obj_type)
		corners = np.array([
			[cx - w / 2.0, cy - h / 2.0],
			[cx + w / 2.0, cy - h / 2.0],
			[cx + w / 2.0, cy + h / 2.0],
			[cx - w / 2.0, cy + h / 2.0],
		])  # from top-left to bottom-left, clock-wise

		# draw box
		corners_order = range(-1, 4)
		axe.plot(corners[corners_order, 0], corners[corners_order, 1], linewidth=2, color=self.colors[obj_type])

		# put a name to it
		text_x, text_y = corners[0, 0] + self.text_offset[0], corners[0, 1] + self.text_offset[1]
		axe.text(text_x, text_y, self.class_names[obj_type],
		         bbox=dict(boxstyle="square", ec=self.colors[obj_type], fc=self.colors[obj_type]))

	def draw_prediction_one_image(self, boxes, axe):
		"""Draw decoded prediction (i.e. boxes) for a single image

		Args:
			boxes (torch.Tensor): shape (nBoxes, 6), each row is (center_x, center_y, w, h, class, score)
			axe (Axes): axe to draw on
		"""
		for i in range(boxes.size(0)):
			if boxes[i, -1] < self.score_threshold:
				# this box has score less than the threshold, so it & the rest are not qualified to be drew
				break
			self.draw_box(boxes[i], axe)

	def draw_prediction(self, batch_boxes, axes):
		"""Draw decoded prediction (i.e. boxes) for a batch of images

		Args:
			batch_boxes list[torch.Tensor]: each tensor has shape (nBoxes, 6),
				each row is (center_x, center_y, w, h, class, score)
			axes (Union[np.ndarray,  Axes]): axes to draw on
		"""
		if isinstance(axes, np.ndarray):
			# batch has >1 images
			assert len(batch_boxes) == axes.size, \
				"axes' shape {} is incompatible with batch size {})".format(axes.shape, len(batch_boxes))
			nCols = axes.shape[1]
			for i in range(axes.size):
				row_idx = i // nCols
				col_idx = i % nCols
				self.draw_prediction_one_image(batch_boxes[i], axes[row_idx, col_idx])
		else:
			# batch has only 1 image
			assert len(batch_boxes) == 1, \
				"Have more than 1 prediction in that batch, so please make axes a numpy.ndarray of Axes"
			self.draw_prediction_one_image(batch_boxes[0], axes)







