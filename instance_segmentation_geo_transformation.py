import numpy as np # dont remove this otherwise gets stuck in infinite loop
import os
from os.path import join
from depth_dataset import ContrastiveDepthDataset
from eval_segmentation import batched_crf
from eval_segmentation import _apply_crf
from eval_segmentation import dense_crf
from src.modules.stego_modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
import maskDepth2 as maskD
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
from utils import get_depth_transform, get_transform
import torchvision.transforms as T
import evaluation_utils as eval_utils
from src.drive_seg.geo_transformations import labelRangeImage

from multiprocessing import Pool, Manager, Process

torch.multiprocessing.set_sharing_strategy('file_system')

def worker(procnum, return_dict, depth_array, mask):
    """worker function"""
    rel_depth = depth_array * mask
    
    return_dict[procnum] = labelRangeImage(rel_depth)

@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "picie"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

    color_list = []
    color_list.append((0, 0, 0))  # 0 values are not part of any instance thus black
    for i in range(1000):
        color = list(np.random.choice(range(256), size=3))
        color_list.append(color)
            
    depth_transform_res = cfg.res

    if cfg.resize_to_original:
        depth_transform_res = cfg.resize_res

    loader_crop = "center"
    image_shape = (depth_transform_res, depth_transform_res)

    test_dataset = ContrastiveDepthDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.experiment_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.res, False, loader_crop),
        target_transform=get_transform(cfg.res, True, loader_crop),
        depth_transform=get_depth_transform(depth_transform_res, loader_crop),
        cfg=model.cfg,
    )

    loader = DataLoader(test_dataset, cfg.batch_size,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    # TODO Try to patch the image into 320x320 and then feed it into the transformer
    with Pool(cfg.num_workers + 5) as pool:
        for i, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                img = batch["img"].cuda()
                label = batch["label"].cuda()
                depth = batch["depth"]
                real_img = batch["real_img"]
                instance = batch["instance"]

                transToImg = T.ToPILImage()
                real_img = real_img[0]
                instance = instance[0].numpy()
                instance = eval_utils.normalize_labels(instance)
                instance_img = grayscale_to_random_color(instance, image_shape, color_list).astype('uint8')
                depth_img = transToImg(depth[0])

                feats, code1 = par_model(img)
                feats, code2 = par_model(img.flip(dims=[3]))
                code = (code1 + code2.flip(dims=[3])) / 2
                code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

                linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
                cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

                #-----------------------------
                # non crf cluster predictions
                #-----------------------------
                
                # cluster_preds = cluster_probs.argmax(1)

                #--------------------------------------------
                # workaround for batch crf as pool.map won't work on my PC
                #--------------------------------------------
                res = []
                for re in map(_apply_crf, zip(img.detach().cpu(), cluster_probs.detach().cpu())):
                    res.append(re)
                
                res = np.array(res)
                cluster_preds = torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in res], dim=0).argmax(1).cuda()

                #--------------
                # batched crf
                #-------------
                # cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()

                model.test_cluster_metrics.update(cluster_preds, label)
                
                tb_metrics = {
                    **model.test_linear_metrics.compute(),
                    **model.test_cluster_metrics.compute(), }

                plotted = model.label_cmap[model.test_cluster_metrics.map_clusters(cluster_preds.cpu())].astype(np.uint8)
                plotted_img = Image.fromarray(plotted[0])
                #plotted_img.show()
                plotted_filtered = filter_classes_has_instance(plotted[0])#sets backgroung to 0
                plotted_img = Image.fromarray(plotted_filtered.astype(np.uint8))
                #plotted_img.show()


                if cfg.resize_to_original:
                    plotted_filtered = resize_mask(plotted_filtered, image_shape)
                    plotted_img = Image.fromarray(plotted_filtered[0].astype(np.uint8))

                masks = maskD.get_segmentation_masks(plotted_img)
                masks.pop(0) # remove the first element which is the mask containing pixels which are classes with no atributtes(e.g. road buildingi)
                masked_depths = maskD.get_masked_depth(depth_img, masks)
                
                depth_array = np.asarray(depth_img)
                depth_array =  np.array(256 * depth_array / 0x0fff, dtype=np.float32)
                
                manager = Manager()
                return_dict = manager.dict()
                jobs = []

                for i in range(len(masks)):
                    p = Process(target=worker, args=(i, return_dict, depth_array, masks[i]))
                    jobs.append(p)
                    p.start()
    
                for proc in jobs:
                    proc.join()

                instance_mask_clustered = np.zeros(depth_array.shape)
                current_num_instances = 0

                # fig, axeslist = plt.subplots(ncols=3, nrows=3)
                for k in return_dict.keys():
                    labels = len(np.unique(return_dict[k])) - 1
                    instance_mask = return_dict[k]
                    
                    instance_mask = np.where(instance_mask != 0, instance_mask + current_num_instances, 0)
                    current_num_instances += labels
                    
                    instance_mask_clustered = np.add(instance_mask_clustered, instance_mask)



                assignments = eval_utils.get_assigment(instance_mask_clustered,
                                                       instance)

                
                instance_mask_pred = np.zeros(image_shape)

                for i, val in enumerate(assignments[1]):
                    mask = np.where(instance_mask_clustered == val, assignments[0][i], 0)
                    instance_mask_pred = instance_mask_pred + mask

                mean_IoU = eval_utils.get_mean_IoU(instance_mask_pred, instance)
                
                    
                print(mean_IoU)

                if mean_IoU < 0.2 or mean_IoU > 0.6:
                    boundingBoxes = eval_utils.get_bounding_boxes(instance_mask_pred).values()
                    tar_boundingBoxes = eval_utils.get_bounding_boxes(instance).values()
                

                    img_boxes = eval_utils.drawBoundingBoxes(real_img.clone().numpy(), boundingBoxes, (0, 255, 0))
                    tar_boxes = eval_utils.drawBoundingBoxes(real_img.clone().numpy(), tar_boundingBoxes, (255, 0, 0))

                    Image.fromarray(grayscale_to_random_color(instance_mask_pred, image_shape, color_list).astype(np.uint8)).save("../results/failure/"+ str(i) + "_" + str(mean_IoU) + "_"  + "preds_cluster.jpeg")
                    Image.fromarray(img_boxes.astype('uint8')).save("../results/failure/" + str(i) + "_" + str(mean_IoU) + "_"  +"preds_bb.jpeg")
                    Image.fromarray(instance_img).save("../results/failure/" + str(i) + "_" + str(mean_IoU) + "_"  +"tar_cluster.jpeg")
                    Image.fromarray(tar_boxes.astype('uint8')).save("../results/failure/" + str(i) + "_" + str(mean_IoU) + "_"  +"tar_bb.jpeg")
                
                f = open("../results/failure/IoU.txt", "a")
                f.write(str(mean_IoU)+" , ")
                f.close()


def filter_classes_has_instance(mask):
    image_shape = mask.shape
    has_instance_list = [
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32)
    ]

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            if not np.any(np.all(mask[i, j] == has_instance_list, axis=1)):
                mask[i, j] = [0, 0, 0]

    return mask


def resize_mask(mask, size):
    mask = torch.tensor(mask.astype('float32'))
    if mask.ndim == 3:
        mask = torch.unsqueeze(mask,0)
    mask = mask.permute((0, 3, 1, 2))

    mask = F.interpolate(input=mask, size=size, mode='bilinear', align_corners=False)
    mask = mask.permute(0, 2, 3, 1)
    mask = mask.numpy()

    plotted_img = Image.fromarray(mask[0].astype(np.uint8))
    # plotted_img.show()

    return mask


def grayscale_to_random_color(grayscale, image_shape, color_list):
    result = np.zeros((image_shape[0], image_shape[1], 3))
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            result[i, j] = color_list[int(grayscale[i, j])]
    return result


if __name__ == "__main__":
    prep_args()
    my_app()
