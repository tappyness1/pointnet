from torch.nn import ReLU, Conv1d, BatchNorm1d, MaxPool1d
import torch.nn as nn
import torch
import numpy as np  

class TransformationNetwork(nn.Module):
    def __init__(self, input_dim:int, output_dim:int) -> None:

        super(TransformationNetwork, self).__init__()
        self.output_dim = output_dim
        self.shared_mlp_1 = nn.Conv1d(input_dim, 64, 1)
        self.batch_norm_1 = nn.BatchNorm1d(64)

        self.shared_mlp_2 = nn.Conv1d(64, 128, 1)
        self.batch_norm_2 = nn.BatchNorm1d(128)

        self.shared_mlp_3 = nn.Conv1d(128, 1024, 1)
        self.batch_norm_3 = nn.BatchNorm1d(1024)

        self.fc_1 = nn.Linear(1024, 512)
        self.batch_norm_4 = nn.BatchNorm1d(512)
        
        self.fc_2 = nn.Linear(512, 256)
        self.batch_norm_5 = nn.BatchNorm1d(256)
        
        self.fc_3 = nn.Linear(256, self.output_dim*self.output_dim)

        self.relu = ReLU()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def forward(self, input)-> torch.Tensor:

        num_points = input.shape[1]
        input = input.transpose(2, 1)

        out = self.relu(self.batch_norm_1(self.shared_mlp_1(input)))
        out = self.relu(self.batch_norm_2(self.shared_mlp_2(out)))
        out = self.relu(self.batch_norm_3(self.shared_mlp_3(out)))

        out = nn.MaxPool1d(num_points)(out)
        out = out.reshape(-1, 1024)

        out = self.relu(self.batch_norm_4(self.fc_1(out)))
        out = self.relu(self.batch_norm_5(self.fc_2(out)))
        out = self.fc_3(out)

        identity_matrix = torch.eye(self.output_dim) # the identity matrix... the OG residual block! 
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.to(self.device)

        out = out.reshape(-1, self.output_dim, self.output_dim) + identity_matrix
        
        return out

class SegmentationNetwork(nn.Module):
    def __init__(self, num_classes=40, input=1088):
        super(SegmentationNetwork, self).__init__()      
        self.shared_mlp_1_1 = nn.Conv1d(input, 512, 1)
        self.shared_mlp_1_2 = nn.Conv1d(512, 256, 1)
        self.shared_mlp_1_3 = nn.Conv1d(256, 128, 1)
        self.batch_norm_1_1 = nn.BatchNorm1d(512)
        self.batch_norm_1_2 = nn.BatchNorm1d(256)
        self.batch_norm_1_3 = nn.BatchNorm1d(128)
        self.relu = ReLU()

        self.shared_mlp_2_1 = nn.Conv1d(128, 128, 1)
        self.batch_norm_2_1 = nn.BatchNorm1d(128)
        self.shared_mlp_2_2 = nn.Conv1d(128, num_classes, 1)
        self.batch_norm_2_2 = nn.BatchNorm1d(num_classes)
        self.relu = ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Feed forward network for segmentations

        Args:
            input (torch.Tensor): b x n x 1088 tensor where n is the number of points in the point cloud space, of the 1088 features, 64 comes from post-feature transformation (local features), 1024 comes from global feature extraction

        Returns:
            torch.Tensor: nxm tensor where n is number of points in the point cloud space and m is the number of classes
        """

        out = input.transpose(2,1)
        out = self.relu(self.batch_norm_1_1(self.shared_mlp_1_1(out)))
        out = self.relu(self.batch_norm_1_2(self.shared_mlp_1_2(out)))
        out = self.relu(self.batch_norm_1_3(self.shared_mlp_1_3(out)))

        out = self.relu(self.batch_norm_2_1(self.shared_mlp_2_1(out)))
        out = self.relu(self.batch_norm_2_2(self.shared_mlp_2_2(out)))
        out = out.transpose(2,1)    

        return out
    
class ClassificationNetWork(nn.Module):
    def __init__(self, num_classes = 40, input=1024):
        super(ClassificationNetWork, self).__init__()
        self.shared_mlp_1_1 = nn.Conv1d(input, 512, 1)
        self.shared_mlp_1_2 = nn.Conv1d(512, 256, 1)
        self.shared_mlp_1_3 = nn.Conv1d(256, num_classes, 1)
        self.batch_norm_1_1 = nn.BatchNorm1d(512)
        self.batch_norm_1_2 = nn.BatchNorm1d(256)
        self.batch_norm_1_3 = nn.BatchNorm1d(num_classes)
        self.relu = ReLU()

    def forward(self, input):
        out = input.transpose(2,1)
        out = self.relu(self.batch_norm_1_1(self.shared_mlp_1_1(out)))
        out = self.relu(self.batch_norm_1_2(self.shared_mlp_1_2(out)))
        out = self.relu(self.batch_norm_1_3(self.shared_mlp_1_3(out))) # do I need relu here? I should probably not softmax the logits yet
        # suppose to dropout somewhere here... but nah
        
        return out

class PointNet(nn.Module):

    def __init__(self, network_type = "Segmentation", num_classes=10):
        super(PointNet, self).__init__()
        self.input_transform = TransformationNetwork(input_dim = 3, output_dim=3)
        self.feature_transform = TransformationNetwork(input_dim = 64, output_dim=64)
        
        # first shared MLP
        self.mlp_1_1 = nn.Conv1d(3,64,1)
        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.mlp_1_2 = nn.Conv1d(64,64,1)

        # second shared MLP
        self.mlp_2_1 = nn.Conv1d(64,64,1)
        self.mlp_2_2 = nn.Conv1d(64,128,1)
        self.mlp_2_3 = nn.Conv1d(128,1024,1)
        self.batch_norm_2_1 = nn.BatchNorm1d(64)
        self.batch_norm_2_2 = nn.BatchNorm1d(128)
        self.batch_norm_2_3 = nn.BatchNorm1d(1024)
        
        self.relu = ReLU()
        self.network_type = network_type

        if network_type == "Segmentation":
            self.segmentation_network = SegmentationNetwork(num_classes=num_classes)
        else: self.classification_network = ClassificationNetWork(num_classes=num_classes)

    def forward(self, input):
        num_points = input.shape[1]
        
        input_transform = self.input_transform(input)
        out = torch.matmul(input, input_transform)

        # first "mlp"
        # can I batch norm using the same batchnorm?
        out = out.transpose(2,1)
        out = self.relu(self.batch_norm_1(self.mlp_1_1(out)))
        out = self.relu(self.batch_norm_1(self.mlp_1_2(out)))
        out = out.transpose(2,1)
        
        # need another one shit
        feature_transform = self.feature_transform(out)
        point_feature_1 = torch.matmul(out, feature_transform) # BxNx64

        # second "mlp"
        out = point_feature_1.transpose(2,1)
        out = self.relu(self.batch_norm_2_1(self.mlp_2_1(out)))
        out = self.relu(self.batch_norm_2_2(self.mlp_2_2(out)))
        out = self.relu(self.batch_norm_2_3(self.mlp_2_3(out)))
        out = nn.MaxPool1d(num_points)(out)
        if self.network_type == "Segmentation":
            # All points have the same "global" features hence need to do some broadcasting to cat it to each point
            # hence our "local" features is BxNx64, but our global features is Bx1X1024
            # we first do a torch repeat on the global features: Bx1x1024 --> BxNx1024
            # now ~kiss~ cat
            point_feature_2 = out.transpose(2,1)
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