import torch
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

        tans = torch.zeros(range_image.shape)
        pdist = torch.nn.PairwiseDistance(p=2)

        for c in range(range_image.shape[0]):
            col_indeces = range_image[c].nonzero()

            if col_indeces.shape[0] <= 1:
                continue

            for i in range(col_indeces.shape[0]):
                if i == 0:
                    continue

                idx_A = col_indeces[i].item()
                idx_B = col_indeces[i-1].item()

                z_A = range_image[c][idx_A]
                z_B = range_image[c][idx_B]

                A = torch.tensor([idx_A, z_A])
                B = torch.tensor([idx_B, z_B])
                C = torch.tensor([idx_A, z_B])

                tans[c][idx_A] = torch.atan2(pdist(B, C), pdist(A, C))

        tans_arr = np.asarray(tans.T)
        plt.imshow(tans_arr)
        plt.show()
        
        labels = torch.zeros(range_image.shape)
        
        for c in range(range_image.shape[0]):
            col_indeces = range_image[c].nonzero()
            
            for y in col_indeces:
                if (labels[c][y[0]] == 0):
                    self.labelGround(c, y[0].item(), labels, tans, range_image)
            
        no_ground = torch.abs(labels - 1) * range_image

        no_ground_arr = np.asarray(no_ground.T)
        plt.imshow(no_ground_arr)
        plt.title("No Ground")
        plt.show()
            
    def labelGround(self, y, x, labels, tans, range_image):
        q = []
        q.append((y, x))
        
        while len(q) > 0:
            p = q[0]
            labels[p] = 1
            
            neighbors = self.neighborhood(p, range_image)
            
            for n in neighbors:
                n = (n[0].item(), n[1].item())
                
                if labels[n] == 1:
                    continue
                
                if np.abs(tans[p] - tans[n]) == 0:
                    q.append(n)
        
            del q[0]
        

    def neighborhood(self, point, range_image, k=2):
        radius = 0
        neighbors = torch.tensor([])

        while len(neighbors) < k+1:
            radius += 1
            
            y_min = max(point[0] - radius, 0)
            y_max = min(point[0] + radius, range_image.shape[0])

            x_min = max(point[1] - radius, 0)
            x_max = min(point[1] + radius, range_image.shape[0])
            
            neighbors = range_image[y_min: y_max, x_min:x_max].nonzero()
            
            for n in neighbors:
                print(n)

        return neighbors
