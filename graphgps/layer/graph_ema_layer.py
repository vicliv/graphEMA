import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter
from torch_geometric.utils import coalesce


def assert_bidirectional(edge_index):
    # Asserts that the edge_index is bidirectional
    # edge_index: (2, E)

    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2

    # NOTE: I'm assuming second half of edge index is the reverse of the first half.
    E = edge_index.shape[1]
    
    assert torch.all(edge_index[0][:E // 2] == edge_index[1][E // 2:])
    assert torch.all(edge_index[1][:E // 2] == edge_index[0][E // 2:])


def make_bidirectional(edge_index):
    # edge_index: (2, E)

    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2

    edge_index, _ = torch.sort(edge_index, 0)
    edge_index = coalesce(edge_index)
    edge_index = torch.cat((edge_index, edge_index[[1, 0]]), dim=1)
    assert_bidirectional(edge_index)
    return edge_index


def expand(x, edge_index):
    # Move features from nodes onto edges
    # x: (N, ...)
    # edge_index: (2, E)

    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2

    return x[edge_index[0]]


def aggregate(M, edge_index):
    # M: (E, d)
    # edge_index: (2, E)

    assert M.dim() == 2
    assert edge_index.dim() == 2
    assert edge_index.shape[1] == M.shape[0]
    assert edge_index.shape[0] == 2
    # check if edge_index[1] contains all the edges
    m = scatter(src=M, index=edge_index[1], dim=0, reduce='sum')
    return m


def propagate(M, m, x, edge_index, deg, lmbda):
    # M: (E, d)
    # m: (N, d)
    # x: (N, d)
    # edge_index: (2, E)
    # deg: (N,)

    assert M.dim() == 2
    assert m.dim() == 2
    assert x.dim() == 2
    # check if x.shape and m.shape is the same if not print the shapes
    assert x.shape == m.shape
    assert edge_index.dim() == 2
    assert deg.dim() == 1
    assert m.shape[1] == M.shape[1]
    assert edge_index.shape[1] == M.shape[0]
    assert edge_index.shape[0] == 2
    assert m.shape[0] == deg.shape[0]

    x_expanded = expand(x, edge_index) # (E, d)
    m_expanded = expand(m, edge_index) # (E, d)
    deg_expanded = expand(deg, edge_index)[..., None] # (E, 1)
        
    # check if deg is zero and print
    # if torch.any((deg_expanded-1) == 0):
    #     print("deg_exp-1 is zero")
    assert_bidirectional(edge_index)
    E = edge_index.shape[1]
    M_ji = torch.cat((M[E // 2:], M[:E // 2]), dim=0)
    epsilon = 1e-3
    M_updated = torch.where(deg_expanded == 1,
        x_expanded,
        (1 - lmbda) * x_expanded + lmbda / (deg_expanded - 1 + 1e-9) * (m_expanded - M_ji))
    # check nan values of M_updated, check if the indices match where deg_expanded ==1
    #print(M_updated)
    if torch.any(torch.isnan(M_updated)):
        print(x)
        print(x_expanded)
        print(M_updated)
        print(lmbda)
        print(1-lmbda)
        print((1 - lmbda) * x_expanded)
        print((deg_expanded - 1) * (m_expanded - M_ji))
        print("nan values in M_updated")
        print(torch.where(torch.isnan(M_updated[:, 0])))
        print("indices where deg_expanded == 1")
        print(torch.where(deg_expanded == 1))
        # print indices where M_updated is the same as x_expanded
        print("indices where M_updated is the same as x_expanded")
        print(torch.where(M_updated == x_expanded))
        #xit()
    return M_updated


def ema(x, edge_index, lmbda, T):
    # Implements one way message passing for graph EMA
    # x: (N, d)
    # edge_index: (2, E)

    assert x.dim() == 2
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2

    E = edge_index.shape[1]
    deg = scatter(src=torch.ones(E, device=x.device), index=edge_index[0], dim=0, reduce='sum')
    M = expand(x, edge_index)
    m = aggregate(M, edge_index)
    if x.shape[0] != m.shape[0]:
        # add zeros to last row of m
        m = torch.cat((m, torch.zeros(x.shape[0] - m.shape[0], m.shape[1], device=m.device)), dim=0)
        deg = torch.cat((deg, torch.zeros(x.shape[0] - deg.shape[0], device=deg.device)), dim=0)
    for i in range(T):
        M = propagate(M, m, x, edge_index, deg, lmbda)
        m = aggregate(M, edge_index)
        if x.shape[0] != m.shape[0]:
            # add zeros to last row of m
            m = torch.cat((m, torch.zeros(x.shape[0] - m.shape[0], m.shape[1], device=m.device)), dim=0)
    ou = torch.where(deg[..., None] == 0, x, (1 - lmbda) * x + lmbda / (deg[..., None]+1e-9) * m)
    return ou


class GraphEMALayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual, lmbda, T):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.lmbda = nn.Parameter(torch.tensor(lmbda)) # NOTE: This could be made into a learnable parameter
        #self.lmbda = lmbda
        self.T = T
        self.model = pyg_nn.Linear(dim_in, dim_out)

    def forward(self, batch):
        x = batch.x
        if self.dim_in == self.dim_out:
            x_in = x.clone()
        x = self.model(x)
        if self.dim_in != self.dim_out:
            x_in = x.clone()
        x = ema(x, batch.edge_index, self.lmbda, self.T)
        # check if x has nan values
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.residual:
            x = x_in + x  # residual connection
        batch.x = x
        return batch
