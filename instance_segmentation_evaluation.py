import sys
import os
from re import T
from shlex import join

import numpy as np
from click.core import F

from utils import get_transform, flexible_collate, prep_args

parentdir = os.path.dirname("../STEGO")
sys.path.append(parentdir)

from multiprocessing import Pool
from src.data.stego_data_utils import ContrastiveSegDataset
from eval_segmentation import batched_crf
from src.modules.stego_modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from src.crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
import src.utils.json_to_binary_mask as Json2BinMask
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
import semantic_to_binary_mask as Seg2BinMask


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
    test_dataset = ContrastiveSegDataset(
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
                polygons = batch["polygons"]

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
                InstanceMasks = Json2BinMask.getBinaryMasks(polygons, cfg.InstanceClasses)

                for className in cfg.InstanceClasses:
                    semanticMask = Semantic2BinMasks.get(className)
                    instances = InstanceMasks[className]
                    for ins in instances:
                        IoU = Json2BinMask.iou(ins, semanticMask)
                        IoU_dict[className][0] += IoU
                        IoU_dict[className][1] += 1
    f = open("../results/predictions/IoU.txt", "a")
    for className in cfg.InstanceClasses:
        f.write(className + "// IoU_sum: " + str(IoU_dict[className][0]) + "   " + "IoU_instance_count: " + str(
            IoU_dict[className][1]) + "\n")
    f.close()


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


if __name__ == "__main__":
    prep_args()
    my_app()
