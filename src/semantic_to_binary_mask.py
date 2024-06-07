import numpy as np

def getMasks(mask, InstanceClasses):
    labels = {
        'unlabeled': [0, 0, 0],
        'ego vehicle': [0, 0, 0],
        'rectification border': [0, 0, 0],
        'out of roi': (0, 0, 0),
        'static': (0, 0, 0),
        'dynamic': [111, 74, 0],
        'ground': [81, 0, 81],
        'road': [128, 64, 128],
        'sidewalk': [244, 35, 232],
        'parking': [250, 170, 160],
        'rail track': [230, 150, 140],
        'building': [70, 70, 70],
        'wall': [102, 102, 156],
        'fence': [190, 153, 153],
        'guard rail': [180, 165, 180],
        'bridge': [150, 100, 100],
        'tunnel': [150, 120, 90],
        'pole': [153, 153, 153],
        'polegroup': [153, 153, 153],
        'traffic light': [250, 170, 30],
        'traffic sign': [220, 220, 0],
        'vegetation': [107, 142, 35],
        'terrain': [152, 251, 152],
        'sky': [70, 130, 180],
        'person': [220, 20, 60],
        'rider': [255, 0, 0],
        'car': [0, 0, 142],
        'truck': [0, 0, 70],
        'bus': [0, 60, 100],
        'caravan': [0, 0, 90],
        'trailer': [0, 0, 110],
        'train': [0, 80, 100],
        'motorcycle': [0, 0, 230],
        'bicycle': [119, 11, 32],
        'license plate': [0, 0, 142],
    }

    binary_masks = {}

    for key in labels:
        if key in InstanceClasses:
            x0 = np.where(mask == labels.get(key), 1, 0).sum(axis=2)
            x = np.where(x0 == 3, 255, 0)
            binary_masks[key] = x

    return binary_masks
