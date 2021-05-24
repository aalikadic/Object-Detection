import numpy as np
import matplotlib.pyplot as plt


def box_compute_cornerNet_radius(box, min_iou=0.3):
    """Compute a radius for building the Gaussian denoting the probability of a pixel being the
        center of thi box

    Args:
         box (list[float]): [xmin, ymin, xmax, ymax]
         min_iou (float): minimum iou of a box, which is defined by two points in two circles around top-left
            & bottom-right corner of the inputted box, with the inputted box. Refer to paper
            CornerNet: Detecting Objects as Paired Keypoints for more detail
    Returns:
        float: radius defined by CornetNet
    """
    width = box[2] - box[0]
    height = box[3] - box[1]

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_iou) / (1 + min_iou)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_iou) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_iou
    b3 = -2 * min_iou * (height + width)
    c3 = (min_iou - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    r = min(r1,r2,r3)
    return r


def compute_iou(box1, box2):
    """Compute iou of two boxes

    Args:
        box1 (list[float]): [xmin, ymin, xmax, ymax]
        box2 (list[float]): [xmin, ymin, xmax, ymax]
    Returns:
        float: iou of box1 & box2
    """
    boxes = np.vstack([box1, box2])  # (2, 4)
    inter_min_xy = np.amax(boxes, axis=0)[:2]  # (2) [xmin, ymin]
    inter_max_xy = np.amin(boxes, axis=0)[2:]  # (2) [xmax, ymax]
    inter_dim = inter_max_xy - inter_min_xy  # (2) [w, h])
    if inter_dim[0] < 0.0 or inter_dim[1] < 0.0:
        return 0.0
    else:
        inter_area = inter_dim[0] * inter_dim[1]
        dims = boxes[:, 2:] - boxes[:, :2]  # (2, 2) [w, h]
        areas = dims[:, 0] * dims[:, 1]  # (2)
        return inter_area / (areas.sum() - inter_area)


def draw_gaussian_on_matrix(box, mat_to_draw_onto, model_downsample_factor=4.0, min_iou=0.3, gaussian_limit=3.0,
                            **kwargs):
    """Splash a box's center onto an image (represented by mat_to_draw_on)

    Args:
        box (list[float]): [xmin, ymin, xmax, ymax]
        mat_to_draw_onto (np.ndarray): shape (H, W), the channel of heat map corresponding the object class
            which box is belong to
        model_downsample_factor (float): CenterNet's total downsample factor, default = 4.0 (as defined by the paper)
        min_iou (float): minimum iou of a box, which is defined by two points in two circles around top-left
            & bottom-right corner of the inputted box, with the inputted box. Refer to paper
            CornerNet: Detecting Objects as Paired Keypoints for more detail
        gaussian_limit (float): gaussian is limited in the radius gaussian_limi * sigma around the center
    """
    # get radius of the gaussian of the box's center
    r = box_compute_cornerNet_radius(box, min_iou)
    sigma = kwargs['sigma'] if 'sigma' in kwargs else r/3.0  # gaussian radius

    # get box's center in the original image
    if 'center' not in kwargs:
        center_x = (box[0] + box[2])/2 # find x-coordiante of box's center
        center_y = (box[1] + box[3])/2  # find y-coordiante of box's center

        center = np.array([center_x, center_y])
    else:
        center = np.array(kwargs['center'])


    p_tilde = np.floor(center/model_downsample_factor)  # (2), center in low-res image

    # get the limitation of the gaussian
    gaussian_xlim = [p_tilde[0] - gaussian_limit * sigma, p_tilde[0] + gaussian_limit * sigma]
    # make sure gaussian_xlim is within image's boundary
    gaussian_xlim = np.clip(gaussian_xlim, 0.0, mat_to_draw_onto.shape[1] - 1.0)


    gaussian_ylim = [p_tilde[1] - gaussian_limit * sigma, p_tilde[1] + gaussian_limit * sigma]
    gaussian_ylim = np.clip(gaussian_ylim, 0.0, mat_to_draw_onto.shape[0] - 1.0)

    # get a grid of integer coordinate with x-coordinate's range defined by gaussian_xlim,
    # and y-coordinate's range defined by gaussian_ylim
    xs = range(int(gaussian_xlim[0]), int(gaussian_xlim[1]) + 1)
    ys = range(int(gaussian_ylim[0]), int(gaussian_ylim[1]) + 1)
    xx, yy = np.meshgrid(xs, ys)

    # compute the probability for every pair of coordinate in (xx, yy)
    pts_in_gaussian = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])  # (n_pts, 2)
    # Note: `pts_in_gaussian` is 2d matrix, each row of which is the (x, y)-coordinate of a point inside the Gaussian

    diff = (pts_in_gaussian - p_tilde) ** 2
    prob = np.exp(-np.sum(diff, axis=1) / (2 * sigma ** 2))  # (n_pts)

    # set the value of pixels in mat_to_draw_onto which are within the box's gaussian to their corresponding prob
    prob_ = np.maximum(prob, mat_to_draw_onto[pts_in_gaussian[:, 1], pts_in_gaussian[:, 0]])
    mat_to_draw_onto[pts_in_gaussian[:, 1], pts_in_gaussian[:, 0]] = prob_
    # Note: the roll of `prob_` is to make sure that if a pixel is inside two different Gaussians, its probability
    # is set by the Gaussian which results in higher value


def box_to_label(box, obj_type, heat_map, offset_map, size_map,
                 model_downsample_factor=4.0, min_iou=0.3, gaussian_limit=3.0):
    """Generate CenterNet's label from an annotated box by adjusting 3 matrices heat_map, offset_map & size_map

    Args:
        box (list[float]): annotated box [xmin, ymin, xmax, ymax]
        obj_type (int): object type the annotated box
        heat_map (np.ndarray): heat map, shape (nClasses, H, W)
        offset_map (np.ndarray): p/R - p_tilda (according to CenterNet paper), shape (2, H, W),
            1st channel is offset for x, 2nd is for y
        size_map (np.ndarray): size map (according to CenterNet paper), shape (2, H, W), (1st channel is w, 2nd is h)
        model_downsample_factor (float): CenterNet's total downsample factor, default = 4.0 (as defined by the paper)
        min_iou (float): minimum iou of a box, which is defined by two points in two circles around top-left
            & bottom-right corner of the inputted box, with the inputted box. Refer to paper
            CornerNet: Detecting Objects as Paired Keypoints for more detail
        gaussian_limit (float): gaussian is limited in the radius gaussian_limi * sigma around the center
    """

    draw_gaussian_on_matrix(box, heat_map[obj_type])


    center_x = box[0]+(box[2]-box[0])/2 #
    center_y = box[1]+(box[3]-box[1])/2  #
    center = np.array([center_x, center_y])  # shape (2)
    center_low_res = np.floor(center/model_downsample_factor).astype(int)  # shape (2) -
    # NOTE: remember to force the type of center_low_res to be int so that you can use it to index into
    # offset_map & size_map


    offset_x = center[0]/model_downsample_factor - center_low_res[0]
    offset_y = center[1]/model_downsample_factor - center_low_res[1]
    offset = np.array([offset_x, offset_y])  # shape (2)
    assert 0.0 <= offset[0] < 1.0 and 0.0 <= offset[1] < 1.0, \
        "Expected offset must be in [0, 1), get {}".format(offset)
    offset_map[0, center_low_res[1], center_low_res[0]] = offset[0]  # x-offset
    offset_map[1, center_low_res[1], center_low_res[0]] = offset[1]  # y-offset


    # Note: remember to divide them for model_downsample_factor
    w, h = (box[2]-box[0])/model_downsample_factor, (box[3]-box[1])/model_downsample_factor
    assert w > 0 and h > 0, "Negative size, w = {}, h = {}".format(w, h)
    size_map[0, center_low_res[1], center_low_res[0]] = w
    size_map[1, center_low_res[1], center_low_res[0]] = h

