import torch
import utils.quad_tree as quad
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes


class nuScenes(Dataset):
    def __init__(self, nusc: NuScenes):
        self.nusc = nusc
        samples = [self.getSamplesForScene(
            scene['first_sample_token'], scene['last_sample_token']) for scene in self.nusc.scene]
        self.samples = [s for sample in samples for s in sample]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]
        cam_front_data = self.nusc.get(
            'sample_data', sample['data']['CAM_FRONT'])
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        pointcloud, colors, image = self.nusc.explorer.map_pointcloud_to_image(
            lidar_data['token'], cam_front_data['token'], )

        transform = transforms.ToTensor()

        image = transform(image)
        pointcloud = torch.tensor(pointcloud)

        meta = {'colors': colors}
        meta['geo-point-seg'] = self.geoPointSegs(pointcloud, colors)

        return pointcloud, image, meta

    def collate_fn(data):
        min = 2**32
        images = []
        depths = []
        batchsize = len(data)

        # To discuss: currently only cutting away, should this be randomized?
        for (d, i) in data:
            min = d.shape[1] if d.shape[1] < min else min

        indices = range(0, min)
        for (d, i) in data:
            images.append(i)
            depths.append(d[:, indices])

        imgs = torch.tensor(
            (batchsize, images[0].shape[0], images[0].shape[1]), dtype=float)
        torch.cat(images, out=imgs)

        dps = torch.tensor(
            (batchsize, depths[0].shape[0], depths[0].shape[1]), dtype=float)
        torch.cat(depths, out=dps)

        return (imgs, dps)

    def getSamplesForScene(self, first_sample_token, last_sample_token, samples=[]):
        if first_sample_token == last_sample_token:
            return samples
        sample = self.nusc.get('sample', first_sample_token)
        samples.append(sample)
        self.getSamplesForScene(sample['next'], last_sample_token, samples)
        return samples

    def geoPointSegs(self, pointcloud, colors):
        pointcloud = torch.round(pointcloud.T)
        pointcloud = self.removeGround(pointcloud, colors)

    def removeGround(self, pointcloud, colors):
        cols = torch.unique(pointcloud.T[0])
        rows = torch.unique(pointcloud.T[1])

        range_image = torch.zeros((cols.shape[0], rows.shape[0]))

        for p in range(pointcloud.shape[0]):
            col = cols == pointcloud[p, 0].item()
            row = rows == pointcloud[p, 1].item()

            x = int(col.nonzero().item())
            y = int(row.nonzero().item())

            range_image[x][y] += colors[p]

        arr = np.asarray(range_image.T)
        plt.imshow(arr)
        plt.show()

        tans = quad.Quad(quad.Point(), quad.Point(
            range_image.shape[0], range_image.shape[1]))
        pdist = torch.nn.PairwiseDistance(p=2)

        for c in range(range_image.shape[0]):
            col_indeces = range_image[c].nonzero()

            if col_indeces.shape[0] <= 1:
                continue

            for i in range(col_indeces.shape[0]):
                idx = col_indeces.shape[0] - i
                if i == 0:
                    tans.insert(quad.Node(quad.Point(idx, c), 0.))
                    continue

                idx_A = col_indeces[idx-1].item()
                idx_B = col_indeces[idx].item()

                z_A = range_image[c][idx_A]
                z_B = range_image[c][idx_B]

                A = torch.tensor([idx_A, z_A])
                B = torch.tensor([idx_B, z_B])
                C = torch.tensor([idx_A, z_B])

                tan = torch.atan2(pdist(B, C), pdist(A, C))

                tans.insert(quad.Node(quad.Point(int(idx_A), c), tan.item()))

        labels = torch.zeros(range_image.shape)

        for c in range(range_image.shape[0]):
            col_indeces = range_image[c].nonzero()

            if (labels[c][col_indeces[-1]] == 0):
                self.labelGround(c, col_indeces[-1].item(), labels, tans)

        no_ground = torch.abs(labels - 1) * range_image

        no_ground_arr = np.asarray(no_ground.T)
        plt.imshow(no_ground_arr)
        plt.title("No Ground")
        plt.show()

    def labelGround(self, y, x, labels, tans: quad.Quad):
        q = []
        q.append(quad.Node(quad.Point(x, y), 0.))

        while len(q) > 0:
            n: quad.Node = q[0]
            labels[n.pos.y][n.pos.x] = 1

            neighbors = self.neighborhood(n.pos, tans)

            for n in neighbors:
                if labels[n.pos.y][n.pos.x] == 1:
                    continue

                if np.abs(n.data - n.data) < 0.0872665 and n not in q:
                    q.append(n)

            q = q[1:]

    def neighborhood(self, point, tans: quad.Quad):
        radius = 20
        neighbors = []

        neighbors = tans.findInRadius(point, radius)

        return neighbors
