from math import ceil

import sys 
sys.path.append('/home/yiya/code_Project_yiya/osa_DGAT/lab/src')
from layer import *
from GAT import GAT,GraphAttentionLayer

class GNNStack(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU(), batch_size=64):

        super().__init__()
        
        # TODO: Sparsity Analysis
        # k_neighs = self.num_nodes = num_nodes
        num_nodes = in_dim
        
        self.num_graphs = groups        
        self.num_feats = seq_len
        self.num_nodes = in_dim
        if seq_len % groups:
            self.num_feats += ( groups - seq_len % groups )
        # self.g_constr = adj_embedding(num_nodes, num_nodes)
        self.g_constr = graph_constructor(num_nodes, num_nodes//2,  num_nodes)
        self.idx = torch.arange(num_nodes)
        
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_dim//2),
            nn.ReLU()
        )
        self.fcn = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_dim//2),
            nn.ReLU()
        )


        # self.cn = nn.Sequential(
        #    nn.Conv1d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm1d(hidden_dim),
        #    nn.ReLU(),
        #    nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm1d(hidden_dim),
        #    nn.ReLU(),
        #    nn.Conv1d(in_channels=hidden_dim, out_channels=out_dim, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm1d(out_dim),
        #    nn.ReLU()
        # )

        
        self.gat = GAT(self.num_feats, hidden_dim,out_dim, dropout,alpha=0.5,nheads=3)
        heads = 1
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'

        paddings = [ (k - 1) // 2 for k in kern_size ]     
        self.left_num_nodes = []
        for layer in range(num_layers + 1):
            left_node = round( num_nodes * (1 - (pool_ratio*layer)) )
            if left_node > 0:
                self.left_num_nodes.append(left_node)
            else:
                self.left_num_nodes.append(1)
        pool_kernel = 3
        self.pool = Dense_TimeDiffPool1d(num_nodes, num_nodes, kernel_size=pool_kernel, padding=(pool_kernel-1)//2)
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.bn = nn.BatchNorm1d(in_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)        
        self.global_pool = nn.AdaptiveMaxPool1d(1)        
        self.linear = nn.Linear(out_dim, num_classes, bias=True)
        
        # self.reset_parameters()
        
        
    def init_adj(self):
        lower_triangle = np.tril(np.ones((self.num_nodes, self.num_nodes)))
        return torch.tensor(lower_triangle, dtype=torch.float32)
        
        
    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        

    def frequen_maxnor(self, x):
        # 对x执行FFT
        x_f = torch.fft.fft(x, dim=-1)
       
        return  torch.cat((x_f.real,x_f.imag), dim=1)
        # return torch.abs(x_f)

        

    def forward(self, inputs: Tensor):
        
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs
        # B, C, N, _ = x.size()
        # x = x.reshape(B, C, N, self.num_graphs, -1)
            
        # adj = self.g_constr(x.device)
        adj = self.g_constr(self.idx.to(x.device),x.device)

        # x_f = self.frequen_maxnor(x)        
        # x = torch.cat((self.tcn(x), self.fcn(x_f)), dim=1)

        x = self.tcn(x)

        x, adj1 = self.pool(self.gat(x,adj), adj)
        # x = self.cn(x)
        
        out = self.global_pool(x.transpose(2,1))
        #out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)

        return out, adj



class DGATStack_v1(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU(), batch_size=64):

        super().__init__()
        
        # TODO: Sparsity Analysis
        # k_neighs = self.num_nodes = num_nodes        
        
        self.num_graphs = groups      
        self.num_nodes = num_nodes*self.num_graphs  
        num_nodes = self.num_nodes
        self.n_feat = seq_len//self.num_graphs       
        self.idx = torch.arange(in_dim)

        # self.cn = nn.Conv1d(in_channels=num_nodes, out_channels=in_dim, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=num_nodes, out_channels=in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU()
        )
        
         # self.g_constr = adj_embedding(num_nodes, num_nodes)
        # self.g_constr = graph_constructor(self.num_nodes, self.num_nodes//2,  self.num_nodes)
        self.g_constr = graph_constructor(in_dim, in_dim//2,  in_dim)
        self.gat = GAT(self.n_feat, hidden_dim,out_dim, dropout,alpha=0.5,nheads=3)
        num_nodes = in_dim
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'

        pool_kernel = 3
        self.pool = Dense_TimeDiffPool1d(num_nodes, num_nodes, kernel_size=pool_kernel, padding=(pool_kernel-1)//2)
        
        # self.num_layers = num_layers
        # self.dropout = dropout
        # self.activation = activation
        # self.bn = nn.BatchNorm1d(in_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)        
        # self.global_pool = nn.AdaptiveMaxPool1d(1)        
        self.linear = nn.Linear(out_dim, num_classes, bias=True)
        
        # self.reset_parameters()
        
        
    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        

    def frequen_maxnor(self, x):
        # 对x执行FFT
        x_f = torch.fft.fft(x, dim=-1)
       
        return  torch.cat((x_f.real,x_f.imag), dim=1)
        # return torch.abs(x_f)
        

    def forward(self, inputs: Tensor):
        
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs
        B, N, C = x.size()
        x = x.reshape(B, N, self.num_graphs, -1)
        x = x.reshape(B, N*self.num_graphs, -1)
            
        # adj = self.g_constr(x.device)
        adj = self.g_constr(self.idx.to(x.device),x.device)
        x = self.tcn(x)
        x, adj1 = self.pool(self.gat(x,adj), adj)
        
        
        # out = self.global_pool(x.transpose(2,1))
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)

        return out, adj



class DGATStack_v2(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, num_nodes, num_classes, dropout=0.5, activation=nn.ReLU(), batch_size=64):

        super().__init__()
        
        # TODO: Sparsity Analysis
        # k_neighs = self.num_nodes = num_nodes        
        
        self.num_graphs = groups      
        self.num_nodes = num_nodes*self.num_graphs  
        num_nodes = self.num_nodes
        self.n_feat = seq_len//self.num_graphs       
        self.idx = torch.arange(num_nodes)

        self.cn = nn.Sequential(
            nn.Conv1d(in_channels=num_nodes, out_channels=out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        
         # self.g_constr = adj_embedding(num_nodes, num_nodes)
        self.g_constr = graph_constructor(self.num_nodes, self.num_nodes//2,  self.num_nodes)
        # self.g_constr = graph_constructor(in_dim, in_dim//2,  in_dim*2)
        self.gat = GAT(self.n_feat, hidden_dim,out_dim, dropout,alpha=0.5,nheads=1)
        # num_nodes = in_dim
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'

        pool_kernel = 3
        self.pool = Dense_TimeDiffPool1d(num_nodes, num_nodes, kernel_size=pool_kernel, padding=(pool_kernel-1)//2)
        
        # self.num_layers = num_layers
        # self.dropout = dropout
        # self.activation = activation
        # self.bn = nn.BatchNorm1d(in_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)        
        self.global_pool = nn.AdaptiveMaxPool1d(1)        
        self.linear = nn.Linear(out_dim, num_classes, bias=True)
        
        # self.reset_parameters()
        
        
    def build_gnn_model(self, model_type):
        if model_type == 'dyGCN2d':
            return DenseGCNConv2d, 1
        if model_type == 'dyGIN2d':
            return DenseGINConv2d, 1
        

    def frequen_maxnor(self, x):
        # 对x执行FFT
        x_f = torch.fft.fft(x, dim=-1)
       
        return  torch.cat((x_f.real,x_f.imag), dim=1)
        # return torch.abs(x_f)

        

    def forward(self, inputs: Tensor):
        
        if inputs.size(-1) % self.num_graphs:
            pad_size = (self.num_graphs - inputs.size(-1) % self.num_graphs) / 2
            x = F.pad(inputs, (int(pad_size), ceil(pad_size)), mode='constant', value=0.0)
        else:
            x = inputs
        B, N, C = x.size()
        x = x.reshape(B, N, self.num_graphs, -1)
        x = x.reshape(B, N*self.num_graphs, -1)
            
        # adj = self.g_constr(x.device)
        adj = self.g_constr(self.idx.to(x.device),x.device)
        x = self.gat(x,adj)
        # x, adj1 = self.pool(x, adj)

        x = self.cn(x)               
        
        # out = self.global_pool(x.transpose(2,1))
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)

        return out, adj


# 纯加入GAT，以输入channel*groups为节点。加入元内相似度邻接矩阵和可学习邻接矩阵
class DGAT(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, N, num_classes, dropout=0.5, activation=nn.ReLU(), batch_size=64):

        super().__init__()
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
    
        self.dropout = dropout
        self.num_graphs = groups      # 切片数
        self.num_nodes = N*self.num_graphs   # 节点数
        self.n_feat = seq_len//self.num_graphs       # 切片后的时序长度
        self.idx = torch.arange(in_dim)

        # self.cn = nn.Conv1d(in_channels=num_nodes, out_channels=in_dim, kernel_size=1)
        topk = self.num_nodes //2  # topk参数

        self.g_constr = graph_constructor_simi(self.num_nodes, N, self.n_feat, groups, batch_size, topk) # 邻接矩阵生成方法
        nheads=1    # 多头数
        alpha=0.5
        self.gat1 = nn.ModuleList(
            [
            GraphAttentionLayer(self.n_feat, nheads*self.n_feat, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nheads*self.n_feat*2, hidden_dim, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(hidden_dim*2, out_dim, dropout=dropout, alpha=alpha, concat=True)
            ]
        )
        self.gat2 = nn.ModuleList(
            [
            GraphAttentionLayer(self.n_feat, nheads*self.n_feat, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nheads*self.n_feat*2, hidden_dim, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(hidden_dim*2, out_dim, dropout=dropout, alpha=alpha, concat=True)
            ]
        )
        
        
        pool_kernel = 3
        # self.pool = Dense_TimeDiffPool1d(num_nodes, num_nodes, kernel_size=pool_kernel, padding=(pool_kernel-1)//2) # 图池化-待处理

        # GRU
    
        
        self.softmax = nn.Softmax(dim=-1)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)        
        self.global_pool = nn.AdaptiveMaxPool1d(1)        
        self.linear = nn.Linear(self.num_nodes, num_classes, bias=True)
        
        # self.reset_parameters()
        

    def forward(self, x: Tensor):
        B, N, C = x.size()
        # 节点生成，切片后扁平化，前按groups划分元内子图
        x = x.reshape(B, N, self.num_graphs, -1)
        x = x.reshape(B, N*self.num_graphs, -1)
            
        # 生成adj，adj1为元内相似度-邻接矩阵，adj2为节点间可学习关系节点-邻接矩阵
        adj1, adj2 = self.g_constr(x)   #,x.device

        for gnet1, gnet2 in zip(self.gat1, self.gat2):
            x1 = gnet1(x, adj1)
            x2 = gnet2(x, adj2)
            x = torch.cat((x1, x2), dim=-1)  
        
        # out = self.global_pool(x.transpose(2,1))
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)

        return out, adj2



# 再DGAT的基础上加入膨胀因果时域卷积
from tcn import *
class MST_DGAT(nn.Module):
    """ The stack layers of GNN.

    """

    def __init__(self, gnn_model_type, num_layers, groups, pool_ratio, kern_size, 
                 in_dim, hidden_dim, out_dim, 
                 seq_len, N, num_classes, dropout=0.5, activation=nn.ReLU(), batch_size=64):

        super().__init__()
        
        assert num_layers >= 1, 'Error: Number of layers is invalid.'
        assert num_layers == len(kern_size), 'Error: Number of kernel_size should equal to number of layers.'
    
        self.dropout = dropout
        self.num_graphs = groups      # 切片数
        self.num_nodes = N*self.num_graphs   # 节点数
        self.n_feat = seq_len//self.num_graphs       # 切片后的时序长度
        self.idx = torch.arange(in_dim)

        # tcn
        topk = self.num_nodes //2  # topk参数

        self.g_constr = graph_constructor_simi(self.num_nodes, N, self.n_feat, groups, batch_size, topk) # 邻接矩阵生成方法
        nheads=1    # 多头数
        alpha=0.5
        self.gat1 = nn.ModuleList(
            [
            GraphAttentionLayer(self.n_feat, nheads*self.n_feat, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nheads*self.n_feat*2, hidden_dim, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(hidden_dim*2, out_dim, dropout=dropout, alpha=alpha, concat=True)
            ]
        )
        self.gat2 = nn.ModuleList(
            [
            GraphAttentionLayer(self.n_feat, nheads*self.n_feat, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(nheads*self.n_feat*2, hidden_dim, dropout=dropout, alpha=alpha, concat=True),
            GraphAttentionLayer(hidden_dim*2, out_dim, dropout=dropout, alpha=alpha, concat=True)
            ]
        )
        
    
        
        self.softmax = nn.Softmax(dim=-1)
        # self.global_pool = nn.AdaptiveAvgPool1d(1)        
        self.global_pool = nn.AdaptiveMaxPool1d(1)        
        self.linear = nn.Linear(self.num_nodes, num_classes, bias=True)
        
        # self.reset_parameters()
        

    def forward(self, x: Tensor):
        B, N, C = x.size()
        # 节点生成，切片后扁平化，前按groups划分元内子图
        x = x.reshape(B, N, self.num_graphs, -1)
        x = x.reshape(B, N*self.num_graphs, -1)
            
        # 生成adj，adj1为元内相似度-邻接矩阵，adj2为节点间可学习关系节点-邻接矩阵
        adj1, adj2 = self.g_constr(x)   #,x.device

        for gnet1, gnet2 in zip(self.gat1, self.gat2):
            x1 = gnet1(x, adj1)
            x2 = gnet2(x, adj2)
            x = torch.cat((x1, x2), dim=-1)  
        
        # out = self.global_pool(x.transpose(2,1))
        out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)

        return out, adj2

