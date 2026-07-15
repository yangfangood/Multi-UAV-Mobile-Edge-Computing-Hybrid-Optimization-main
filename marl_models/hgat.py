import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, global_mean_pool
import config

class HeteroGAT(nn.Module):
    """
    异构图注意力网络，用于处理多智能体之间的异构关系。
    节点类型：当前只有 'agent' 一种（但可通过扩展支持 'user', 'base_station' 等）。
    边类型：'agent_to_agent'。
    实际上，由于我们只有一种节点类型，这退化为普通的GAT。为了体现异构性，我们可以在特征中加入类型编码。
    更复杂的情况可扩展。
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # 定义异构图卷积：只处理 agent 到 agent 的边
            conv = HeteroConv({
                ('agent', 'to', 'agent'): GATConv(in_dim, hidden_dim, heads=num_heads, concat=False, dropout=config.HGAT_DROPOUT),
            }, aggr='sum')
            self.convs.append(conv)
            in_dim = hidden_dim
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: {'agent': (num_nodes, in_dim)}
        edge_index_dict: {('agent', 'to', 'agent'): (2, E)}
        """
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        # 只输出 agent 节点的特征
        return x_dict['agent']