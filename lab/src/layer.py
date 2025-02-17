import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np

class multi_shallow_embedding(nn.Module):
    
    def __init__(self, num_nodes, k_neighs, num_graphs):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.k = k_neighs
        self.num_graphs = num_graphs

        self.emb_s = Parameter(Tensor(num_graphs, num_nodes, 1))
        self.emb_t = Parameter(Tensor(num_graphs, 1, num_nodes))
        # self.reset_parameters()
        
    def reset_parameters(self):
        init.xavier_uniform_(self.emb_s)
        init.xavier_uniform_(self.emb_t)
        
        
    def forward(self, device):
        
        # adj: [G, N, N]
        adj = torch.matmul(self.emb_s, self.emb_t).to(device)
        
        # remove self-loop
        adj = adj.clone()
        idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        adj[:, idx, idx] = float('-inf')
        
        # top-k-edge adj
        adj_flat = adj.reshape(self.num_graphs, -1)
        indices = adj_flat.topk(k=self.k)[1].reshape(-1)
        
        idx = torch.tensor([ i//self.k for i in range(indices.size(0)) ], device=device)
        
        adj_flat = torch.zeros_like(adj_flat).clone()
        adj_flat[idx, indices] = 1.
        adj = adj_flat.reshape_as(adj)
        
        return adj



class adj_embedding(nn.Module):
    
    def __init__(self, num_nodes, k_neighs):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.k = k_neighs

        self.emb_s = Parameter(Tensor(num_nodes, 1))
        self.emb_t = Parameter(Tensor(1, num_nodes))
        # self.adj = Parameter(self.init_adj())
        self.reset_parameters()


    def reset_parameters(self):
        # init.xavier_uniform_(self.emb_s)
        # init.xavier_uniform_(self.emb_t)
        init.uniform_(self.emb_s)
        init.uniform_(self.emb_t)
        
    def forward(self, device):
        
        # adj: [N, N]
        adj = F.relu(torch.matmul(self.emb_s, self.emb_t).to(device))
        # adj = self.adj
        
        # # remove self-loop
        adj = adj.clone()
        # idx = torch.arange(self.num_nodes, dtype=torch.long, device=device)
        # adj[:, idx] = float('-inf')
        
        # # top-k-edge adj
        # adj_flat = adj.reshape(self.num_graphs, -1)
        s1, indices = adj.topk(k=self.k//2)
        
        # idx = torch.tensor([ i for i in range(indices.size(0)) ], device=device)
        adj_flat = adj.clone()
        adj_flat = torch.zeros_like(adj).to(device=device)
        adj_flat.fill_(float('0'))
        # for i in indices[0]:
        #     adj_flat[i, indices[i, :]]=1
        adj_flat.scatter_(1, indices,s1.fill_(1))
        # adj = adj_flat.reshape_as(adj)
        adj=adj*adj_flat
        return adj


# 图结构-链接矩阵生成-idx
class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        #self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx, device):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class graph_constructor_simi(nn.Module):
    def __init__(self, nnodes, N, length, group, batch_size, k):
        super(graph_constructor_simi, self).__init__()
        assert nnodes==group*N , 'Error:  nnodes!=group*N'
        self.nnodes = nnodes
        self.group = group
        self.n = N
        self.k = k

        self.W = nn.Parameter(torch.randn(length, nnodes))  # [d, L]
        self.b = nn.Parameter(torch.zeros(nnodes))            # [d]
        
        # 相似度计算参数
        self.alpha = nn.Parameter(torch.tensor([0.1]))          # 缩放因子
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def compute_node_similarity(self, x: torch.Tensor, method: str = "cosine") -> torch.Tensor:
        """
        计算批处理时间序列节点间的相似度矩阵
        Args:
            x (Tensor): 输入张量，形状为 [B, N, L]
            method (str): 相似度计算方法，可选 "cosine" | "pearson" | "euclidean"
        Returns:
            Tensor: 相似度矩阵，形状为 [B, N, N]
        """
        assert method in ["cosine", "pearson", "euclidean"], "不支持的相似度计算方法"

        if method in ["cosine", "pearson"]:
            # 中心化处理（皮尔逊相关系数需要）
            if method == "pearson":
                x = x - x.mean(dim=-1, keepdim=True)
            
            # 归一化
            x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8) # 防止除零
            
            # 计算相似度矩阵
            similarity = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

        elif method == "euclidean":
            # 计算欧氏距离的平方
            x_square = torch.sum(x**2, dim=-1, keepdim=True)  # [B, N, 1]
            similarity = x_square + x_square.transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
            
            # 将距离转换为相似度（可选：exp(-distance)）
            # similarity = torch.exp(-similarity)  # 根据需求调整

        return similarity

    
    def set_topk(self, A, dim=-1):
        values, indices = torch.topk(A, k=self.k, dim=dim)
        mask = torch.zeros_like(A)
        mask.scatter_(dim=dim, index=indices, src=torch.ones_like(values))
        return A * mask


    def forward(self, x):
        A_intra = torch.zeros((x.shape[0], self.nnodes, self.nnodes)).to(x.device)
        for start in range(0, self.nnodes, self.group):
            end = start + self.group
            A_intra[:, start:end, start:end] = self.compute_node_similarity(x[:, start:end, :])


        # [B, N, L] -> [B, N, l] 通过线性投影
        h = torch.einsum('bnl,lk->bnk', x, self.W) + self.b  # 应用W和b
        h = torch.relu(h)  # 非线性激活
        h_norm = h / (torch.norm(h, dim=-1, keepdim=True) + 1e-6)
        sim_matrix = torch.einsum('bnd,bmd->bnm', h_norm, h_norm)  # [B, N, N]
        A_inter = torch.sigmoid(self.alpha * sim_matrix)  # 映射到[0,1]

        # A_intra = self.set_topk(A_intra)
        A_inter = self.set_topk(A_inter)
        
        return A_intra, A_inter 




class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        #self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx, device):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class Group_Linear(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=False):
        super().__init__()
                
        self.out_channels = out_channels
        self.groups = groups
        
        self.group_mlp = nn.Conv2d(in_channels * groups, out_channels * groups, kernel_size=(1, 1), groups=groups, bias=bias)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.group_mlp.reset_parameters()
        
        
    def forward(self, x: Tensor, is_reshape: False):
        """
        Args:
            x (Tensor): [B, C, N, F] (if not is_reshape), [B, C, G, N, F//G] (if is_reshape)
        """
        B = x.size(0)
        C = x.size(1)
        N = x.size(-2)
        G = self.groups
        
        if not is_reshape:
            # x: [B, C_in, G, N, F//G]
            x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        # x: [B, G*C_in, N, F//G]
        x = x.transpose(1, 2).reshape(B, G*C, N, -1)
        
        out = self.group_mlp(x)
        out = out.reshape(B, G, self.out_channels, N, -1).transpose(1, 2)
        
        # out: [B, C_out, G, N, F//G]
        return out


class DenseGCNConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, bias=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.lin = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        self.lin.reset_parameters()
        init.zeros_(self.bias)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[:, idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [B, G, N, N]
        """
        adj = self.norm(adj, add_loop).unsqueeze(1)

        # x: [B, C, G, N, F//G]
        x = self.lin(x, False)
        
        out = torch.matmul(adj, x)
        
        # out: [B, C, N, F]
        B, C, _, N, _ = out.size()
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        if self.bias is not None:
            out = out.transpose(1, -1) + self.bias
            out = out.transpose(1, -1)
        
        return out


class DenseGINConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, groups=1, eps=0, train_eps=True):
        super().__init__()
        
        # TODO: Multi-layer model
        self.mlp = Group_Linear(in_channels, out_channels, groups, bias=False)
        
        self.init_eps = eps
        if train_eps:
            self.eps = Parameter(Tensor([eps]))
        else:
            self.register_buffer('eps', Tensor([eps]))
            
        self.reset_parameters()
            
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.init_eps)
        
    def norm(self, adj: Tensor, add_loop):
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(adj.size(-1), dtype=torch.long, device=adj.device)
            adj[..., idx, idx] += 1
        
        deg_inv_sqrt = adj.sum(-1).clamp(min=1).pow(-0.5)
        
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
        return adj
        
        
    def forward(self, x: Tensor, adj: Tensor, add_loop=True):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        B, C, N, _ = x.size()
        G = adj.size(0)
        
        # adj-norm
        adj = self.norm(adj, add_loop=False)
        
        # x: [B, C, G, N, F//G]
        x = x.reshape(B, C, N, G, -1).transpose(2, 3)
        
        out = torch.matmul(adj, x)
        
        # DYNAMIC
        x_pre = x[:, :, :-1, ...]
        
        # out = x[:, :, 1:, ...] + x_pre
        out[:, :, 1:, ...] = out[:, :, 1:, ...] + x_pre
        # out = torch.cat( [x[:, :, 0, ...].unsqueeze(2), out], dim=2 )
        
        if add_loop:
            out = (1 + self.eps) * x + out
        
        # out: [B, C, G, N, F//G]
        out = self.mlp(out, True)
        
        # out: [B, C, N, F]
        C = out.size(1)
        out = out.transpose(2, 3).reshape(B, C, N, -1)
        
        return out


class Dense_TimeDiffPool2d(nn.Module):
    
    def __init__(self, pre_nodes, pooled_nodes, kern_size, padding):
        super().__init__()
        
        # TODO: add Normalization
        self.time_conv = nn.Conv2d(pre_nodes, pooled_nodes, (1, kern_size), padding=(0, padding))
        
        self.re_param = Parameter(Tensor(kern_size, 1))
        
    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')
        
        
    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, C, N, F]
            adj (Tensor): [G, N, N]
        """
        x = x.transpose(1, 2)
        out = self.time_conv(x)
        out = out.transpose(1, 2)
        
        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        
        return out, out_adj



class Dense_TimeDiffPool1d(nn.Module):
    
    def __init__(self, pre_nodes, pooled_nodes, kernel_size, padding):
        super().__init__()
        
        # TODO: add Normalization
        self.time_conv = torch.nn.Conv1d(pre_nodes, pooled_nodes, kernel_size, stride=1, padding=padding, dilation=1, groups=1)

        
        self.re_param = Parameter(Tensor(kernel_size, 1))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.time_conv.reset_parameters()
        init.kaiming_uniform_(self.re_param, nonlinearity='relu')
        
        
    def forward(self, x: Tensor, adj: Tensor):
        """
        Args:
            x (Tensor): [B, N, F]
            adj (Tensor): [N, N]
        """
        # x = x.transpose(1, 2)
        out = self.time_conv(x)
        # out = out.transpose(1, 2)
        
        # s: [ N^(l+1), N^l, 1, K ]
        s = torch.matmul(self.time_conv.weight, self.re_param).view(out.size(-2), -1)

        # TODO: fully-connect, how to decrease time complexity
        out_adj = torch.matmul(torch.matmul(s, adj), s.transpose(0, 1))
        
        return out,out_adj
