from math import ceil

import sys 
sys.path.append('/home/yiya/code_Project_yiya/osa_DGAT/lab/src')
from layer import *
from GAT import GAT

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
        # self.g_constr = adj_embedding(num_nodes, num_nodes)
        self.g_constr = graph_constructor(self.num_nodes, self.num_nodes//2,  self.num_nodes)
        self.idx = torch.arange(self.num_nodes)
        
        self.gat = GAT(self.n_feat, hidden_dim,out_dim, dropout,alpha=0.5,nheads=3)
        
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
        B, N, C = x.size()
        x = x.reshape(B, N, self.num_graphs, -1)
        x = x.reshape(B, N*self.num_graphs, -1)
            
        # adj = self.g_constr(x.device)
        adj = self.g_constr(self.idx.to(x.device),x.device)
        x, adj1 = self.pool(self.gat(x,adj), adj)
        # x = self.cn(x)
        
        out = self.global_pool(x.transpose(2,1))
        #out = self.global_pool(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        #out = self.softmax(out)

        return out, adj
