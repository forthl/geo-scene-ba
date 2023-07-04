import hydra
import os
import src.pointnet2.provider as provider

from omegaconf import DictConfig
from os.path import join


from src.pointnet2.pointnet2_part_seg_msg import get_model

from src.data.depth_dataset import ContrastiveDepthDataset
from src.modules.stego_modules import *

from src.utils.data_utils import generatePointCloudFromRangeImage, imageFromPointsAndClassification, project_disparity_to_3d

seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}


def to_categorical(y_len, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.ones((1, 1, 16))
    return new_y.cuda()


def main():
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat
    
    classifier = get_model(50, normal_channel=False)
    checkpoint = torch.load('./checkpoints/saved_models/pointnet2/pointnet_no_normals.pth')
    img_path = "./tmp_data/aachen_000000_000019_gtFine_color.png"
    depth_path = "./tmp_data/aachen_000000_000019_disparity_small.png"
    
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.cuda()
    classifier = classifier.eval()
    
    depth_array = np.asarray(Image.open(depth_path))
    # points = generatePointCloudFromRangeImage(depth_array)
    points = project_disparity_to_3d(depth_path)

    points = torch.unsqueeze(points.cuda(), 0)
    
    vote_pool = torch.zeros((117265, 50)).cuda()
        
    for _ in range(2):
        seg_pred, _ = classifier(points, to_categorical(1, 16))
        seg_pred = seg_pred.contiguous().view(-1, 50)
        vote_pool += seg_pred
        
    seg_pred = vote_pool / 2
    cur_pred_val = np.argmax(seg_pred[:points.shape[2]].cpu().data.numpy(), axis=-1)
    # cur_pred_val_logits = cur_pred_val
    # cur_pred_val = np.zeros((1, points.shape[2])).astype(np.int32)
    # 
    # logits = cur_pred_val_logits[0, :]
    # cur_pred_val[0, :] = np.argmax(seg_pred.cpu().data.numpy(), axis=-1)
    
    imageFromPointsAndClassification(points, cur_pred_val, depth_array)

if __name__ == '__main__':
    main()