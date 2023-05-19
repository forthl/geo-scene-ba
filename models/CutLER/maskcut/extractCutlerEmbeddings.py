# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Mostly copy-paste from CutLER repo. https://github.com/facebookresearch/CutLER
"""

import os
import sys

sys.path.append('../')

import numpy as np
import dino
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy.linalg import eigh
from scipy import ndimage
import json

from third_party.TokenCut.unsupervised_saliency_detection import utils, metric
from third_party.TokenCut.unsupervised_saliency_detection.object_discovery import detect_box

# crf codes are are modfied based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py
from crf import densecrf

# Image transformation applied to all images
ToTensor = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(
                                (0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),])

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = F.normalize(feats, p=2, dim=0)
    A = (feats.transpose(0,1) @ feats).cpu().numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix
    dim, num_patch = feats.size()[0], feats.size()[1]
    painting = painting + mask.unsqueeze(0)
    painting[painting > 0] = 1
    painting[painting <= 0] = 0
    feats = feats.clone().view(dim, ps, ps)
    feats = ((1 - painting) * feats).view(dim, num_patch)
    return feats, painting

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def maskcut_forward(feats, dims, scales, init_image_size, tau=0, N=3, cpu=False):
    """
        Implementation of MaskCut.
        Inputs
          feats: the pixel/patche features of an image
          dims: dimension of the map from which the features are used
          scales: from image to map scale
          init_image_size: size of the image
          tau: thresold for graph construction
          N: number of pseudo-masks per image.
        """
    bipartitions = []
    eigvecs = []

    for i in range(N):
        if i == 0:
            painting = torch.from_numpy(np.zeros(dims))
            if not cpu: painting = painting.cuda()
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)

        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)

        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dims)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)

        # get pxiels corresponding to the seed
        bipartition = bipartition.reshape(dims).astype(float)
        _, _, _, cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dims)
        pseudo_mask[cc[0], cc[1]] = 1
        pseudo_mask = torch.from_numpy(pseudo_mask)
        if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = torch.sum(pseudo_mask) / pseudo_mask.size()[0] / pseudo_mask.size()[1]
            if metric.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dims)
                pseudo_mask = torch.from_numpy(pseudo_mask)
                if not cpu: pseudo_mask = pseudo_mask.to('cuda')
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = F.interpolate(pseudo_mask.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        bipartition_masked = bipartition.cpu().numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dims)
        eigvec = torch.from_numpy(eigvec)
        if not cpu: eigvec = eigvec.to('cuda')
        eigvec = F.interpolate(eigvec.unsqueeze(0).unsqueeze(0), size=init_image_size, mode='nearest').squeeze()
        eigvecs.append(eigvec.cpu().numpy())

    return seed, bipartitions, eigvecs

def maskcut(img_path, backbone,patch_size, tau, N=1, fixed_size=480, cpu=False):
    I = Image.open(img_path).convert('RGB')
    bipartitions, eigvecs = [], []

    I_new = I.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
    I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, patch_size)

    tensor = ToTensor(I_resize).unsqueeze(0)
    feat = backbone(tensor)[0]

    _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h, w], tau, N, cpu)

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs, I_new

def post_process_dcrf(I, bipartition, cpu):
    # post-process pesudo-masks with CRF
    pseudo_mask = densecrf(np.array(I_new), bipartition)
    pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)

    # filter out the mask that have a very different pseudo-mask after the CRF
    mask1 = torch.from_numpy(bipartition)
    mask2 = torch.from_numpy(pseudo_mask)
    #
    if not cpu:
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
    if metric.IoU(mask1, mask2) < 0.5:
        pseudo_mask = pseudo_mask * -1
        print("Check")

    # construct binary pseudo-masks
    pseudo_mask[pseudo_mask < 0] = 0

    return pseudo_mask

def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)

def create_image_info(image_id, file_name, image_size,
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
    }
    return image_info


def create_annotation_info(annotation_id, image_id, bipartition, crop, feat):
    # Return annotation info

    upper = np.max(bipartition)
    lower = np.min(bipartition)
    thresh = upper / 2.0
    bipartition[bipartition > thresh] = upper
    bipartition[bipartition <= thresh] = lower

    binary_mask_encoded = mask.encode(np.asfortranarray(bipartition.astype(np.uint8)))
    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    rle = mask_util.encode(np.array(bipartition[..., None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "bbox": crop,
        "area": area.tolist(),
        # "features": feat.tolist(),
        "segmentation": segmentation
    }

    return annotation_info

output = {
    "images": [],
    "annotations": []
}

url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
feat_dim = 768
vit_arch = 'base'
vit_feat = 'k'
patch_size = 8
img_path = 'imgs/test.jpg'
fixed_size = 120
out_dir = 'out'
tau = 0.2
N = 6

if out_dir is not None and not os.path.exists(out_dir):
    os.mkdir(out_dir)

backbone = dino.ViTFeat(url, feat_dim, vit_arch, vit_feat, patch_size)
backbone.eval()
feat = ''

image_id, segmentation_id = 1, 1

try:
    bipartitions, eigvecs, I_new = maskcut(img_path, backbone, patch_size, tau=tau, N=N, fixed_size=fixed_size, cpu=True)
except:
    print(f'Skipping {img_path}')

img_name = img_path

I = Image.open(img_path).convert('RGB')
width, height = I.size
crop_list = []  # store crops in form of left, top, right and bottom bounds
feat_list = []
image_names = []

for pseudo_mask in bipartitions:

    # post process and filter out masks
    #pseudo_mask = post_process_dcrf(I_new, pseudo_mask, False)

    # find bounding box or continue if it doesn't exist
    if np.any(pseudo_mask):
        rmin, rmax, cmin, cmax = bbox2(pseudo_mask)
    else:
        print("Pseudo mask is empty")
        continue

    # crop original image to bounding box size
    rmin = height * rmin / fixed_size
    rmax = height * rmax / fixed_size
    cmin = width * cmin / fixed_size
    cmax = width * cmax / fixed_size
    im1 = I.crop((cmin, rmin, cmax, rmax))
    crop = (cmin, rmin, cmax, rmax)

    # get DINO features for cropped object
    I_new = im1.resize((fixed_size, fixed_size), PIL.Image.LANCZOS)
    I_resize, _, _, _, _ = utils.resize_pil(I_new, patch_size)
    tensor = ToTensor(I_resize).unsqueeze(0)
    feat = backbone(tensor)[0].numpy()


    # create image info
    if img_name not in image_names:
        image_info = create_image_info(image_id, img_path, (height, width, 3))
        output["images"].append(image_info)
        image_names.append(img_name)

    # create annotation info
    annotation_info = create_annotation_info(segmentation_id, image_id, pseudo_mask, crop, feat)
    if annotation_info is not None:
        output["annotations"].append(annotation_info)
        segmentation_id += 1


for crop in crop_list:
    im1 = I.crop(crop)
    im1.show()

# save annotations
json_name = '{}/cityscapes_fixsize{}_tau{}_N{}.json'.format(out_dir, fixed_size, tau, N)

with open(json_name, 'w') as output_json_file:
    json.dump(output, output_json_file)