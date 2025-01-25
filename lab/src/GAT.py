"""
@FIile      : GAT.py
@Copyright  :
@Author     :yiya_luo
@Function   :
@Date       :2023/10/24
@Description：
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm



class GraphAttentionLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.heads = 1
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.W)
        init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        B, N, infeature = x.size()

        
        h = torch.matmul(x, self.W)  # B ,N ,O
        # 生成N*N的嵌入,repeat两次，两种方式，这个地方应该是为了把不同的sample拼接。为下一步求不同样本交互做准备。
        # a_input = torch.cat([h.repeat(1,1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=1).view(B, N, -1, 2 * self.out_features)
        a_input = torch.cat([h.unsqueeze(2).expand(-1, -1, N, -1), h.unsqueeze(1).expand(-1, N, -1, -1)], dim=-1)
        # [N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        # [N, N, 1] => [N, N] 图注意力的相关系数（未归一化）

        zero_vec = -1e12 * torch.ones_like(e)  # 将没有连接的边置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N, N]
        # attention = adj*e
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留，
        # 否则需要mask并置为非常小的值，原因是softmax的时候这个最小值会不考虑。
        attention = F.softmax(attention, dim=1)  # softmax形状保持不变 [N, N]，得到归一化的注意力权重！
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout，防止过拟合
        h_prime = torch.matmul(attention, h)  # [N, N].[N, out_features] => [N, out_features]
        # 得到由周围节点通过注意力权重进行更新的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, ofeat, dropout, alpha, nheads):
        '''
        nfeat 是输入特征的维度。
        nhid 是隐藏层特征的维度。
        ofeat 是输出特征的维度。
        dropout 是 Dropout 概率，用于控制模型的过拟合。
        alpha 是 LeakyReLU 的负斜率参数。
        nheads 是注意力头的数量
        '''
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.Sequential(
            GraphAttentionLayer(nfeat, nheads*nfeat, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nheads*nfeat, nhid, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nhid, ofeat, dropout=dropout, alpha=alpha, concat=True)
        )
 

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        for att in self.attentions:
            x = att(x,adj)

        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_att(x, adj))
        # return F.log_softmax(x, dim=1)
        return x

