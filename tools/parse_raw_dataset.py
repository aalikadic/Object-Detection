import json
import os
import xml.etree.ElementTree as ET
import sys


class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

class_to_int = dict(zip(class_names, range(len(class_names))))
# get full path to the root directory of this project
proj_root = os.path.dirname(os.path.abspath(__file__)).split(os.sep)
proj_root = os.path.join(os.sep, *proj_root[:-1])
proj_root = proj_root[:2] + '\\' + proj_root[2:]

def parse_annotation(ann_file, model_input_size):
    """Parse VOC's annotation file

    Args:
        ann_file (str): full path to an annotation file
        model_input_size (tuple[float]): (W, H) of CenterNet's input
    Returns:
        dict: {'boxes': [xmin, ymin, xmax, ymax], 'labels': [int]}
    """
    tree = ET.parse(ann_file)
    root = tree.getroot()

    # get image size
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    # compute downsample factor to resize image from its original shape to model_input_size
    downsample_w = float(width) / float(model_input_size[0])
    downsample_h = float(height) / float(model_input_size[1])

    boxes = list()
    labels = list()
    for obj in root.iter('object'):
        label = obj.find('name').text.lower().strip()
        if label not in class_names:
            # skip objects whose labels are not in VOC'07
            continue

        bbox = obj.find('bndbox')
        xmin = (float(bbox.find('xmin').text) - 1) / downsample_w
        ymin = (float(bbox.find('ymin').text) - 1) / downsample_h
        xmax = (float(bbox.find('xmax').text) - 1) / downsample_w
        ymax = (float(bbox.find('ymax').text) - 1) / downsample_h

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(class_to_int[label])

    return {'boxes': boxes, 'labels': labels}


def create_data_lists(dataset_path, model_input_size=(384, 384)):
    """Merge VOC'07 & '12 into two lists. One lists store full path of every image in these datasets
        The other store a list of dicts, each dict is the annotation for the image at the index. The result
        is saved to file in project_root/data. This function assume the following folder structure:
        dataset_path
        |___VOC2007
            |___Annotations
            |___ImageSets
        |___VOC2012
            |___Annotations

    Args:
        dataset_path (str): full path to folder containing VOC2007 & VOC20012
        model_input_size (tuple[float]): (W, H) of CenterNet's input
    """
    subset = ['VOC2007', 'VOC2012']

    # training set = VOC'07 trainval + VOC'12 trainval
    training_image_path = list()
    training_annotations = list()
    nObjects = 0  # to count the total number of objects
    for dataset in subset:
        print('Parsing trainval of ', dataset, '...')
        # get the list of trainval images
        with open(os.path.join(dataset_path, dataset, 'ImageSets', 'Main', 'trainval.txt'), 'r') as f:
            images_list = f.readlines()
        # clean up images' names by removing the trailing '\n'
        images_list = [im.strip() for im in images_list]
        images_list.sort()
        # for each image in images_list, create its full path & parse its annotation
        for image in images_list:
            image_path = os.path.join(dataset_path, dataset, 'JPEGImages', image+'.jpg')
            annotation = parse_annotation(os.path.join(dataset_path, dataset, 'Annotations', image+'.xml'),
                                          model_input_size)
            nObjects += len(annotation['boxes'])
            # store image_path & annotation in their corresponding lists
            training_image_path.append(image_path)
            training_annotations.append(annotation)

    # dump training data into hard disk
    with open(os.path.join(proj_root, 'data', 'train_images.json'), 'w') as f:
        json.dump(training_image_path, f)

    with open(os.path.join(proj_root, 'data', 'train_annotations.json'), 'w') as f:
        json.dump(training_annotations, f)

    print('There are {} training images containing the total of {} objects'.format(
        len(training_image_path), nObjects
    ))


    print("Parsing test set from VOC2007's test set")
    test_image_path, test_annotations = list(), list()
    nObjects = 0
    # get the list of trainval images
    with open(os.path.join(dataset_path, 'VOC2007', 'ImageSets', 'Main', 'test.txt'), 'r') as f:
        images_list = f.readlines()
    # clean up images' names by removing the trailing '\n'
    images_list = [im.strip() for im in images_list]
    images_list.sort()

    # for each image in images_list, create its full path & parse its annotation
    for image in images_list:
        image_path = os.path.join(dataset_path, 'VOC2007', 'JPEGImages', image+'.jpg')
        annotation = parse_annotation(os.path.join(dataset_path, 'VOC2007', 'Annotations', image+'.xml'),
                                          model_input_size)
        nObjects += len(annotation['boxes'])
        # store image_path & annotation in their corresponding lists
        test_image_path.append(image_path)
        test_annotations.append(annotation)

    # dump testing data into hard disk
    with open(os.path.join(proj_root, 'data', 'test_images.json'), 'w') as f:
        json.dump(test_image_path, f)

    with open(os.path.join(proj_root, 'data', 'test_annotations.json'), 'w') as f:
        json.dump(test_annotations, f)

    print('There are {} test images containing the total of {} objects'.format(
        len(test_image_path), nObjects
    ))


if __name__ == '__main__':
    if len(sys.argv) not in (2, 3):
        print("Usage: python parse_raw_dataset.py full_path_to_directory_containing_voc2007_voc2012 [model_input_size]")
        exit(0)
    voc_root = sys.argv[1]
    model_input_size = sys.argv[2] if len(sys.argv) == 3 else 384
    create_data_lists(voc_root, tuple([model_input_size]*2))
