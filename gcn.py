import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib as plt


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        conv of each node in each time step
        refer to https://github.com/FelixOpolka/STGCN-PyTorch/blob/master/stgcn.py
        :param in_channels: Number of input features at each node in each time_step.
        :param out_channels: Desired number of output channels at each node in each time_step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes, num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class BasicGcn(nn.Module):
    def __init__(self, node_size, input_szie, out_szie):
        '''
        :param A: adjustment matrix
        :param input_szie: input dim of node feature
        :param out_szie: output dim of node feature
        '''
        super(BasicGcn, self).__init__()
        self.node_num = node_size
        self.weight = nn.Parameter(torch.FloatTensor(node_size, node_size))
        self.feature_weight = nn.Parameter(torch.FloatTensor(input_szie, out_szie))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        stdv1 = 1. / math.sqrt(self.feature_weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.feature_weight.data.uniform_(-stdv1, stdv1)

    def get_normalized_adj(self,A):
        """
        Returns the degree normalized adjacency matrix.
        """
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
        D = np.array(np.sum(A, axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5  # Prevent infs
        diag = np.reciprocal(np.sqrt(D))
        A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                             diag.reshape((1, -1)))
        return A_wave

    def forward(self, X, A):
        A_hat = self.get_normalized_adj(A)
        adj = self.weight*A_hat
        feature = torch.einsum("jkli,ij->jklj", [X, self.feature_weight])
        # X size (batch_size, num_nodes, num_time_steps,input_feature_dim)
        # feature = torch.mul(X, self.feature_weight)
        # out=torch.mm(adj,feature)
        out=torch.einsum("ij,jklm->kilm", [adj, feature.permute(1, 0, 2, 3)])
        '''
        feature_input size (num_nodes,batch_size, num_time_steps, output_feature_dim)
        out size (batch_size,num_nodes,num_time_steps, output_feature_dim)
        '''
        return F.relu(out), adj


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.gcn = BasicGcn(num_nodes, in_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        t2, adj = self.gcn(A_hat, t)
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3), adj


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4


if __name__ == '__main__':
    A=np.eye(8)
    # model= BasicGcn(A, 3, 16)
    model2 = STGCN(9, 3, 15, 1)
    for name, parameter in model2.named_parameters():
        print(name, ':', parameter.size())
    parameters = sum(param.numel() for param in model2.parameters())
    print('total parameter is {}'.format(parameters))


