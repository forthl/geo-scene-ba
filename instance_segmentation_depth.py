import hydra
import os
import random
import torch.multiprocessing

import maskDepth2 as maskD
import numpy as np
import torchvision.transforms as T

from multiprocessing import Pool
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from eval_segmentation import batched_crf
from src.crf import  dense_crf
from src.data.depth_dataset import ContrastiveDepthDataset
from src.modules.stego_modules import *
import src.utils.semantic_to_binary_mask as Seg2BinMask
from train_segmentation import LitUnsupervisedSegmenter

torch.multiprocessing.set_sharing_strategy('file_system')


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = join(root)
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(join(self.root, self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


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

    loader_crop = "center"
    test_dataset = ContrastiveDepthDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.experiment_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(cfg.res, False, loader_crop),
        target_transform=get_transform(cfg.res, True, loader_crop),
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

    IoU_dict = {}
    for className in cfg.InstanceClasses:
        IoU_dict[className] = [0, 0]  # [IoUSum, IoUInstances]

    with Pool(cfg.num_workers + 5) as pool:
        for i, batch in enumerate(tqdm(loader)):
            with torch.no_grad():
                img = batch["img"].cuda()
                label = batch["label"].cuda()
                depth = batch["depth"]


                trans_unormalize = T.Compose([ T.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                T.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               T.ToPILImage()])

                transToImg= T.ToPILImage()

                depth_img = transToImg(depth)
                #label_img = transToImg(label[0].cpu())
                #real_img =trans_unormalize(img[0].cpu())
                #depth_img.show()
                #label_img.show()
                #real_img.show()

                feats, code1 = par_model(img)
                feats, code2 = par_model(img.flip(dims=[3]))
                code = (code1 + code2.flip(dims=[3])) / 2

                code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

                linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
                cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

                linear_crf = batched_crf(pool, img, linear_probs).argmax(1).cuda()
                cluster_crf = batched_crf(pool, img, cluster_probs).argmax(1).cuda()

                model.test_cluster_metrics.update(cluster_crf, label)

                tb_metrics = {
                    **model.test_linear_metrics.compute(),
                    **model.test_cluster_metrics.compute(),
                }

                plotted = model.label_cmap[model.test_cluster_metrics.map_clusters(cluster_crf.cpu())].astype(np.uint8)

                Semantic2BinMasks = Seg2BinMask.getMasks(plotted[0], cfg.InstanceClasses)

                plotted_img = Image.fromarray(plotted[0])
                plotted_img.show()

                semantic_mask_car = [transToImg(Semantic2BinMasks.get('car'))]

                masks = maskD.get_segmentation_masks(plotted_img)

                masked_depths = maskD.get_masked_depth(depth_img, masks)
                # save_masks(masked_depths)
                point_clouds = maskD.create_point_clouds(masked_depths)

                result = np.zeros((320,320))
                current_num_instances=0
                # K-Means Test
                for point_cloud in point_clouds:
                    data = np.transpose(point_cloud)
                    max_k = 10  # Maximum value of k to consider
                    if point_cloud.shape[1]==0:  #check if its an empty point cloud. Look into this bug later
                        continue
                    optimal_k = maskD.find_optimal_k(data, min(max_k,point_cloud.shape[1]),'vrc')
                    #print(f"Optimal value of k: {optimal_k}")
                    #labels, centroids, instance_mask = maskD.kmeans_clustering(data, optimal_k,current_num_instances)
                    labels, instance_mask = maskD.spectral_clustering(data, optimal_k, current_num_instances)
                    result=np.add(result, instance_mask)
                    current_num_instances+=optimal_k
                    #print("Cluster labels:", labels)
                    #print("Centroids:", centroids)
                    #maskD.visualize_clusters(data, labels, centroids)
                    #print('a')
                result=grayscale_to_random_color(result,current_num_instances).astype(np.uint8)
                Image.fromarray(result).convert('RGB').show()
                print(result)


def grayscale_to_random_color(grayscale,num_colors):
    color_list=[]
    for i in range(num_colors):
        color = list(np.random.choice(range(256), size=3))
        color_list.append(color)
    result = np.zeros((320,320,3))
    for i in  range(319):
        for j in range(319):
            result[i,j]=color_list[int(grayscale[i,j])]
    return result



def get_trans(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper])


def get_transform1(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])

if __name__ == "__main__":
    prep_args()
    my_app()
