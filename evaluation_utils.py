import cv2
import numpy as np
import torch.multiprocessing
from scipy.optimize import linear_sum_assignment


def get_assigment(preds,
                  instance):  # there might be less predicted instances than actual ones so make it that it returns correct result

    preds = normalize_labels(preds)
    instance = normalize_labels(instance)

    preds = torch.tensor(preds, dtype=torch.int32)
    instance = torch.tensor(instance, dtype=torch.int32)

    num_predicted_instances = int(torch.max(preds).numpy()) + 1
    num_actual_instances = int(torch.max(instance).numpy()) + 1

    if num_predicted_instances > num_actual_instances:
        cost_matrix = torch.zeros((num_actual_instances, num_predicted_instances), dtype=torch.int32)

        preds = preds.reshape(-1)
        instance = instance.reshape(-1)

        total_num = (torch.tensor(num_predicted_instances)) * instance + preds

        cost_matrix = torch.bincount(
            total_num,
            minlength=num_actual_instances * num_predicted_instances)
        cost_matrix = cost_matrix.reshape((num_actual_instances, num_predicted_instances))

        assigments = linear_sum_assignment(cost_matrix, maximize=True)

    else:
        cost_matrix = torch.zeros((num_predicted_instances, num_actual_instances), dtype=torch.int32)

        preds = preds.reshape(-1)
        instance = instance.reshape(-1)

        total_num = (torch.tensor(num_actual_instances)) * preds + instance

        cost_matrix = torch.bincount(
            total_num,
            minlength=num_actual_instances * num_predicted_instances)
        cost_matrix = cost_matrix.reshape((num_predicted_instances, num_actual_instances))
        assigments = linear_sum_assignment(cost_matrix, maximize=True)
        assigments = np.flip(assigments)

    return assigments


def normalize_labels(instances):
    label_ids = np.unique(instances)
    current_id = 0

    for label_id in label_ids:
        instances = np.where(instances == label_id, current_id, instances)
        current_id += 1

    return instances


def get_bounding_boxes(img):
    instance_ids = np.unique(img)
    instance_ids = np.delete(instance_ids, 0)
    bounding_boxes = {}

    for id in instance_ids:  # value 0 are object which have no instances
        indexes = np.nonzero(np.where(img == id, img, 0))
        bounding_boxes[id] = (np.max(indexes[1]), np.max(indexes[0]), np.min(indexes[1]),
                              np.min(indexes[0]))  # Bounding box encoded as (Right, Bottom, Left, Top)

    return bounding_boxes


def bb_intersection_over_union(boxA, boxB):  # bounding boxes are in array-like coordinate system
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = min(boxA[0], boxB[0])
    yA = min(boxA[1], boxB[1])
    xB = max(boxA[2], boxB[2])
    yB = max(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xA - xB, 0)) * max((yA - yB), 0))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[0] - boxA[2]) * (boxA[1] - boxA[3]))
    boxBArea = abs((boxB[0] - boxB[2]) * (boxB[1] - boxB[3]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_mean_IoU(preds, target):
    bounding_boxes_target = get_bounding_boxes(target)
    bounding_boxes_preds = get_bounding_boxes(preds)
    IoU = []

    for index in list(bounding_boxes_target.keys()):
        if bounding_boxes_preds.get(index) is not None:
            IoU.append(bb_intersection_over_union(bounding_boxes_target[index], bounding_boxes_preds[index]))
        else:
            IoU.append(0)

    mean_IoU = sum(IoU) / len(IoU)

    return mean_IoU



def drawBoundingBoxes(imageData, inferenceResults, color):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    for res in inferenceResults:
        right = int(res[0])
        bottom = int(res[1])
        left = int(res[2])
        top = int(res[3])
        imgHeight, imgWidth, _ = imageData.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)

    return imageData