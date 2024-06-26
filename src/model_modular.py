from torch.nn import ReLU, Conv1d, BatchNorm1d, MaxPool1d
import torch.nn as nn
import torch
import numpy as np

class TNetMLP(nn.Module):
    def __init__(self, input_dim, out_dim) -> None:
        super().__init__()

        self.shared_mlp = nn.Conv1d(input_dim, out_dim, 1)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.relu = ReLU()

    def forward(self, input) -> torch.Tensor:
        # assume input is already transposed/permuted
        out = self.relu(self.batch_norm(self.shared_mlp(input)))
        return out
    
class TNetFC(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = ReLU()

    def forward(self, input) -> torch.Tensor:
        out = self.relu(self.batch_norm(self.fc(input)))
        return out
    
class TransformationNetwork(nn.Module):
    def __init__(self, input_dim:int, output_dim:int) -> None:
        super(TransformationNetwork, self).__init__()

        # instantiate the TNetMLP objects.
        # your input_dim can be 3 or any feature
        self.mlp_1 = TNetMLP(input_dim,64)
        self.mlp_2 = TNetMLP(64,128)
        self.mlp_3 = TNetMLP(128,1024)

        self.fc_1 = TNetFC(1024, 512)
        self.fc_2 = TNetFC(512, 256)

        self.output_dim = output_dim
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input)-> torch.Tensor:
        num_points = input.shape[1]

        input = input.transpose(2, 1)
        out = self.mlp_1(input)
        out = self.mlp_2(out)
        out = self.mlp_3(out)

        out = nn.MaxPool1d(num_points)(out)
        out = out.reshape(-1, 1024)

        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.to(self.device)

        out = out.reshape(-1, self.output_dim, self.output_dim) + identity_matrix

        return out
    
class PointNetBackBone(nn.Module):
    def __init__(self):
        super().__init__()

        # Tranformation Network
        self.input_transform = TransformationNetwork(input_dim = 3, output_dim=3)
        self.feature_transform = TransformationNetwork(input_dim = 64, output_dim=64)

        # first shared MLP
        self.mlp_1_1 = TNetMLP(3, 64)
        self.mlp_1_2 = TNetMLP(64, 64)

        # second shared MLP
        self.mlp_2_1 = TNetMLP(64, 64)
        self.mlp_2_2 = TNetMLP(64, 128)
        self.mlp_2_3 = TNetMLP(128, 1024)


    def forward(self, input):

        num_points = input.shape[1]

        # input transformation
        input_transform = self.input_transform(input)
        out = torch.matmul(input, input_transform)

        # first mlp
        out = out.transpose(2,1)
        out = self.mlp_1_1(out)
        out = self.mlp_1_2(out)
        out = out.transpose(2,1)

        # feature transformation
        feature_transform = self.feature_transform(out)
        point_feature_1 = torch.matmul(out, feature_transform) # BxNx64

        # second MLP. Remember to transpose point_feature_1
        out = point_feature_1.transpose(2,1)
        out = self.mlp_2_1(out)
        out = self.mlp_2_2(out)
        out = self.mlp_2_3(out)

        # maxpool
        out = nn.MaxPool1d(num_points)(out)

        point_feature_2 = out.transpose(2,1)

        return point_feature_1, point_feature_2
    
class SegmentationNetwork(nn.Module):
    def __init__(self, num_classes=4, input=1088):
        super(SegmentationNetwork, self).__init__()

        self.mlp_1_1 = TNetMLP(input, 512)
        self.mlp_1_2 = TNetMLP(512, 256)
        self.mlp_1_3 = TNetMLP(256, 128)

        self.mlp_2_1 = TNetMLP(128, 128)
        self.mlp_2_2 = TNetMLP(128, num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        out = input.transpose(2,1)
        out = self.mlp_1_1(out)
        out = self.mlp_1_2(out)
        out = self.mlp_1_3(out)

        out = self.mlp_2_1(out)
        out = self.mlp_2_2(out)
        out = out.transpose(2,1)

        return out
    
class ClassificationNetWork(nn.Module):
    def __init__(self, num_classes = 40, input=1024):
        super(ClassificationNetWork, self).__init__()
        self.mlp_1_1 = TNetMLP(input, 512)
        self.mlp_1_2 = TNetMLP(512, 256)
        self.mlp_1_3 = TNetMLP(256, num_classes)

    def forward(self, input):
        out = input.transpose(2,1)
        out = self.mlp_1_1(out)
        out = self.mlp_1_2(out)
        out = self.mlp_1_3(out)

        # suppose to dropout somewhere here... but nah
        return out
    
class PointNet(nn.Module):

    def __init__(self, network_type="Classification", num_classes=10):
        super(PointNet, self).__init__()

        self.backbone = PointNetBackBone()
        if network_type == "Segmentation":
            self.segmentation_network = SegmentationNetwork(num_classes=num_classes)
        else: self.classification_network = ClassificationNetWork(num_classes=num_classes)
        self.network_type = network_type

    def forward(self, input):
        num_points = input.shape[1]
        point_feature_1, point_feature_2 = self.backbone(input)

        if self.network_type == "Segmentation":
            # All points have the same "global" features hence need to do some broadcasting to cat it to each point
            # hence our "local" features is BxNx64, but our global features is Bx1X1024
            # we first do a torch repeat on the global features: Bx1x1024 --> BxNx1024
            # now ~kiss~ cat
            point_feature_2 = point_feature_2.repeat(1,num_points,1)
            segmentation_input = torch.cat((point_feature_1, point_feature_2), dim = 2)
            out = self.segmentation_network(segmentation_input)
        else:
            # need to transpose the out before running it through the classification head.
            out = out.transpose(2,1)
            out = self.classification_network(out)

        return out

if __name__ == "__main__":

    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 2500, 3).astype('float32')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X).to(device)

    model = PointNet(network_type = "Segmentation", num_classes=4)
    # model = PointNet(network_type = "Classification", num_classes=10)
    model = model.to(device)
    
    summary(model, (1000,3))
    print ()
    print (model.forward(X).shape)