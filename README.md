# PointNet

## What this is about
Just a simple implementation based on the PointNet which was in 2017 a revolutionary way of handling point cloud data, which was to... directly look at PointClouds.

## What has been done 

1. Set up the Architecture - See /src/model.py
1. Set up the dataset and dataloader - See /src/data_preprocessing folder
1. Set up the training - see /src/train.py
1. Set up validation - Outputs classification reports
1. Results visualisation - check out ./notebooks/segmentation_inference.ipynb

## Dataset Used

### Classification - ModelNet

The famous ModelNet dataset. The classification network was trained on ModelNet10.

### Part Segmentation - ShapeNet

The segmentation model was trained on the Airplane class in ShapeNet. Do note that the annotation is pretty good, and the parts annotated for one class is not the same as the next class. For example, the Airplane class has parts 0 - 3. The next class would have parts 4 and 5. So there is no overlap.  

## How to run 

Make sure you change the directory of your data.

```
python -m src.train
```

## Useful Sources

1. [Paper itself](https://arxiv.org/abs/1612.00593)
1. [Tensorflow Implementation](https://github.com/charlesq34/pointnet/tree/master) - makes sense because it was published by Google DeepMind and they were all in on TF back in the day. Used Conv2d in place of Conv1D
1. [Conv1D vs Linear Layer](https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer#comment97851680_55576314) - talks about the difference between a Conv1D and Linear Layer (assume no strides etc). Honestly, there doesn't seem to be a difference besides some rounding issues
1. [Sample PyTorch implementation](https://colab.research.google.com/drive/12RQDCV7krZtfjwJ0B4bOEBnvnDHTu-k2?usp=sharing#scrollTo=ycw_6xYaHiyf) - a good starting point to check PyTorch implementation. The issue is that some of the layers are different from the paper. Can probably also get away with using reshape instead of this view thing
1. [Some explanation on PointNet](https://www.digitalnuage.com/pointnet-or-the-first-neural-network-to-handle-directly-3d-point-clouds) - an OK explanation but skims the STN part which is fine.