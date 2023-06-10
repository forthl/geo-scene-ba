import numpy as np
import PIL.Image as Image

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val) * 255
    return normalized_arr

# get masks from segmentations
def get_segmentation_masks(img_path):
    masks = []
    I = Image.open(img_path).convert('L')
    I = np.asarray(I)
    for c in np.unique(I):
        segmentation = I == c
        # test = np.uint8(segmentation * 255)
        masks.append(segmentation)
    return masks

# mask depth image with segmentations
def get_masked_depth(depth_path, masks):
    D = Image.open(depth_path)
    depth_array = np.asarray(D)
    depth_array = normalize_array(depth_array)
    masked_depths = []
    for mask in masks:
        seg_masked = np.where(mask, depth_array, 0)
        masked_depth = np.uint8(seg_masked)
        masked_depth = normalize_array(masked_depth)
        masked_depths.append(masked_depth)
        # masked_depth = Image.fromarray(masked_depth)
        # masked_depth.show()
    return masked_depths


def save_masks(masked_depths):
    for i, d in enumerate(masked_depths):
        masked_depth = Image.fromarray(d).convert('RGB')
        masked_depth.save("aachen_000000_000019_disparity_mask_" + str(i) + ".jpg")

def create_point_clouds(masked_depths):
    point_clouds = []
    for mask in masked_depths:
        non_zero = np.nonzero(mask)
        point_cloud = np.array([non_zero[0], non_zero[1], mask[non_zero[0],non_zero[1]]])
        point_clouds.append(point_cloud)
    return point_clouds

if __name__ == '__main__':
    img_path = "aachen_000000_000019_gtFine_color.png"
    depth_path = "aachen_000000_000019_disparity.png"

    masks = get_segmentation_masks(img_path)
    masked_depths = get_masked_depth(depth_path, masks)
    #save_masks(masked_depths)
    point_clouds = create_point_clouds(masked_depths)



