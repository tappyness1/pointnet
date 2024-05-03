import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
import matplotlib.pyplot as plt
import os
import pandas as pd
from src.data_processing.dataset_utils import farthest_point_sample, pc_normalize


class Model40NetDataset(Dataset):
    def __init__(self, cfg):
        self.data_path = cfg['data_path']
        if cfg['train']:
            self.dataset_path = self.data_path + '/modelnet40_train.txt'
        else:
            self.dataset_path = self.data_path + '/modelnet40_test.txt'
        
        self.dataset = [line.rstrip() for line in open(self.dataset_path)]

        shape_names = [line.rstrip() for line in open(self.data_path + '/modelnet40_shape_names.txt')]
        self.classes = dict(zip(shape_names, range(len(shape_names))))
        # self.classes = {shape: i for i, shape in enumerate(shape_names)}
        
    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx):

        item_name = self.dataset[idx].split('_')
        class_name = item_name[0] if len(item_name) == 2 else '_'.join(item_name[:-1])
        point_cloud_path = os.path.join(self.data_path,class_name, f"{self.dataset[idx]}.txt")
        point_clouds = pd.read_csv(point_cloud_path, header = None).to_numpy().astype('float32')
        point_clouds = farthest_point_sample(point_clouds, 1024) # can use torch3d but this is fine too
        point_clouds = pc_normalize(point_clouds)
        class_id = self.classes[class_name]

        return point_clouds[:, :3], class_id

if __name__ == "__main__":
    cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False}
    model40net_dataset = Model40NetDataset(cfg)
    dataloader = torch.utils.data.DataLoader(model40net_dataset, batch_size=4, shuffle=True)
    for i, (point_clouds, class_id) in enumerate(dataloader):
        print (point_clouds.shape, class_id.shape)
        break
