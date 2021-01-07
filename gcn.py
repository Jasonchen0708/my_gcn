import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib as plt


class BasicGcn(nn.Module):
    def __init__(self, A, input_szie, out_szie):
        '''
        :param A: adjustment matrix
        :param input_szie: input dim of node feature
        :param out_szie: output dim of node feature
        '''
        super(BasicGcn, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(A.shape[0], A.shape[1]))
        self.feature_weight = nn.Parameter(torch.FloatTensor(input_szie, out_szie))
        self.A_norm = self.get_normalized_adj()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        stdv1 = 1. / math.sqrt(self.feature_weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.feature_weight.data.uniform_(-stdv1, stdv1)

    def get_normalized_adj(A):
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
        A_hat=self.A_norm(A)
        adj = self.weight*A_hat
        feature = torch.mul(X,self.feature_weight)
        out=torch.mm(adj,feature)
        return F.relu(out), adj

class LSTM(nn.Module):
    def __init__(self):
        super (LSTM, self).__init__()


if __name__ == '__main__':
    A=np.eye(8)
    model= BasicGcn(A, 3, 16)
    for name, parameter in model.named_parameters():
        print(name, ':', parameter.size())


