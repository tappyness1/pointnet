import torch
from torch.utils.data import DataLoader, Subset
# from torchsummary import summary
from tqdm import tqdm
import numpy as np
from src.model import PointNet
from src.data_processing.dataset import ModelNetDataset
from src.metrics import process_confusion_matrix
from src.validation import validation

import torch.nn as nn
import torch.optim as optim


def train_classifier(train_set, val_set, cfg, num_classes = 40):

    loss_function = nn.CrossEntropyLoss() # using energy_loss instead
    
    # network = UNet(img_size = 572, num_classes = num_classes + 1)
    network = PointNet(network_type="classifier", num_classes = num_classes)

    network.train()

    # optimizer = optim.Adam(network.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    optimizer = optim.RMSprop(network.parameters(),
                            lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'], momentum=cfg['train']['weight_decay'], foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)

    # if cfg['show_model_summary']:
    #     summary(network, (1024, 3))

    if cfg['train']['subset']:
        subset_indices = torch.randperm(len(train_set))[:cfg['train']['subset']]
        train_set = Subset(train_set, subset_indices)
    
    train_dataloader = DataLoader(train_set, batch_size=6, shuffle = True)

    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        with tqdm(train_dataloader) as tepoch:
            for imgs, labels in tepoch:
                # print (imgs.shape)
                # print (labels)

                optimizer.zero_grad() 
                out = network(imgs.to(device))
                loss = loss_function(out, labels.unsqueeze(1).to(device))
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
        
        loss = validation(network, val_set, cfg, get_metrics = False)
        scheduler.step(loss)
        network.train()
        
    print("training done")
    torch.save(network, cfg['save_model_path'])

    print("Validating dataset")
    validation(network, val_set, cfg, get_metrics = True)

    return network

if __name__ == "__main__":

    torch.manual_seed(42)

    cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": True, "modelnet_type": "modelnet10",}
    train_set = ModelNetDataset(cfg)
    cfg = {"data_path": "../data/modelnet40_normal_resampled", "train": False, "modelnet_type": "modelnet10",}
    val_set = ModelNetDataset(cfg)

    cfg = {"save_model_path": "model_weights/model_weights.pt",
           'show_model_summary': True, 
           'train': {"epochs": 10, 'lr': 1e-4, 
                     'weight_decay': 1e-8, 'momentum':0.999, 
                     'subset': False, # set False if not intending to use subset. Set to 20 or something for small dataset experimentation/debugging
                     'num_classes': 10} # ModelNet40 so 40 classes, whereas ModelNet10 so 10 classes
            }
    train_classifier(train_set = train_set, val_set = val_set,  cfg = cfg, num_classes = cfg['train']['num_classes'])
