# PointNet

## What this is about
Just a simple implementation based on the PointNet which was in 2017 a revolutionary way of handling point cloud data, which was to... directly look at PointClouds. This is just to get a better understanding of the PointNet and I won't be doing training here. Will be handled when I reach its final(as at 2024 don't flame me) form as in ResPointNet++.

## What has been done 

1. Set up the Architecture

## What else needs to be done

1. Set up the dataset and dataloader
1. Set up the training.
1. Set up validation, but only takes accuracy and loss. 
1. Results visualisation

## How to run 

Make sure you change the directory of your data. I used PSNet (not yet implemented yet)

```
python -m src.main
```

## Useful Sources

1. [Paper itself](https://arxiv.org/abs/1612.00593)
1. [Tensorflow Implementation](https://github.com/charlesq34/pointnet/tree/master) - makes sense because it was published by Google DeepMind and they were all in on TF back in the day. Used Conv2d in place of Conv1D
1. [Conv1D vs Linear Layer](https://stackoverflow.com/questions/55576314/conv1d-with-kernel-size-1-vs-linear-layer#comment97851680_55576314) - talks about the difference between a Conv1D and Linear Layer (assume no strides etc). Honestly, there doesn't seem to be a difference besides some rounding issues
1. [Sample PyTorch implementation](https://colab.research.google.com/drive/12RQDCV7krZtfjwJ0B4bOEBnvnDHTu-k2?usp=sharing#scrollTo=ycw_6xYaHiyf) - a good starting point to check PyTorch implementation. The issue is that some of the layers are different from the paper. Can probably also get away with using reshape instead of this view thing
1. [Some explanation on PointNet](https://www.digitalnuage.com/pointnet-or-the-first-neural-network-to-handle-directly-3d-point-clouds) - an OK explanation but skims the STN part which is fine.