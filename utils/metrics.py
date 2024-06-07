import numpy as np


def iou(y_true, y_pred, num_classes):
    iou_list = []
    for c in range(num_classes):
        true_c = y_true == c
        pred_c = y_pred == c
        intersection = np.logical_and(true_c, pred_c)
        union = np.logical_or(true_c, pred_c)
        iou = np.sum(intersection) / np.sum(union)
        iou_list.append(iou)
    return iou_list


def miou(y_true, y_pred, num_classes):
    iou_list = iou(y_true, y_pred, num_classes)
    miou = np.mean(iou_list)
    return miou, iou_list


def instance_iou(y_true, y_pred):
    instances = np.unique(y_true)
    instance_iou_list = []
    for i in instances:
        if i == 0: # ignore background class
            continue
        true_i = y_true == i
        pred_i = y_pred == i
        intersection = np.logical_and(true_i, pred_i)
        union = np.logical_or(true_i, pred_i)
        iou = np.sum(intersection) / np.sum(union)
        instance_iou_list.append(iou)
    return instance_iou_list


def instance_miou(y_true, y_pred):
    instance_iou_list = instance_iou(y_true, y_pred)
    instance_miou = np.mean(instance_iou_list)
    return instance_miou, instance_iou_list
