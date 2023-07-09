import multiprocessing
import numpy as np

from maskDepth import get_segmentation_masks
from PIL import Image

import matplotlib.pyplot as plt

from src.drive_seg.geo_transformations import removeGround, labelRangeImage, find_NNs
from src.utils.representation import plotGraph


def worker(procnum, return_dict, depth_array, mask):
    """worker function"""
    rel_depth = depth_array * mask
    
    return_dict[procnum] = labelRangeImage(rel_depth)

def main():
    img_path = "./tmp_data/aachen_000000_000019_gtFine_color.png"
    depth_path = "./tmp_data/aachen_000000_000019_disparity.png"
    
    masks = get_segmentation_masks(img_path)
            
    image = Image.open(depth_path)
    depth_array = np.asarray(image)
    depth_array =  np.array(256 * depth_array / 0x0fff, dtype=np.float32)
    # plotGraph(depth_array)
    # _, ground_mask = removeGround(depth_array)
    # plotGraph(depth_array)
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    
    for i in range(len(masks)):
        p = multiprocessing.Process(target=worker, args=(i, return_dict, depth_array, masks[i]))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()
        
    array = np.zeros(depth_array.shape)
    # plotGraph(array)
    
    fig, axeslist = plt.subplots(ncols=4, nrows=4)
    for k in return_dict.keys():
        array += return_dict[k]
        axeslist.ravel()[k].imshow(return_dict[k])
        axeslist.ravel()[k].set_title(k)
        axeslist.ravel()[k].set_axis_off()
    axeslist.ravel()[14].imshow(array)
    plt.tight_layout() # optional
    
    plt.show()

    print('Yippie')

if __name__ == '__main__':
    main()