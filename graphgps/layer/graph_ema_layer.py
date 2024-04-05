import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_scatter import scatter
from torch_geometric.utils import coalesce
import math

class ApplyLambda(nn.Module):
    def __init__(self, D: int = 300, dt_max: float = 0.1, dt_min: float = 0.001):
        super(ApplyLambda, self).__init__()  # Corrected to use the correct class name
        
        # Initialize parameters of the discretization
        log_dt = (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        
        log_lambda_real = torch.log(0.5 * torch.ones(D // 2))
        lambda_imag = torch.arange(D // 2) * math.pi
        
        self.register_parameter("log_dt", nn.Parameter(torch.tensor(log_dt)))
        self.register_parameter("log_lambda_real", nn.Parameter(log_lambda_real))
        self.register_parameter("lambda_imag", nn.Parameter(lambda_imag))

    def forward(self, x: torch.Tensor, one_minus=False):  # Corrected type hint
        dt = torch.exp(self.log_dt)
        lambda_complex = torch.exp((-torch.exp(self.log_lambda_real) + 1j * self.lambda_imag) * dt)
        
        # Get magnitude and phase directly
        lambda_magnitude = 1 - torch.abs(lambda_complex) if one_minus else torch.abs(lambda_complex)
        lambda_phase = torch.angle(lambda_complex)
        
        # Assuming x is [N, D], reshape it to work with complex pairs
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        
        # Apply rotation and scaling
        x_rotated_real = (torch.cos(lambda_phase) * x_pairs[..., 0] - torch.sin(lambda_phase) * x_pairs[..., 1]) * lambda_magnitude
        x_rotated_imag = (torch.sin(lambda_phase) * x_pairs[..., 0] + torch.cos(lambda_phase) * x_pairs[..., 1]) * lambda_magnitude
        
        # Combine real and imaginary parts back to a tensor of the original shape
        x_transformed = torch.stack((x_rotated_real, x_rotated_imag), -1).view_as(x)
        
        return x_transformed

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out) -> None:
        super().__init__()
        hidden_dim = (dim_in + dim_out)

        self.fc1 = nn.Linear(dim_in, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim_in, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.fc1(x)) * self.fc2(x)
        x = self.proj(x)
        return x
    
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

    M_updated = torch.where(deg_expanded == 1,
        x_expanded,
        lmbda(x_expanded, True) + lmbda(m_expanded - M_ji) / (deg_expanded - 1 + 1e-9))
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
    return torch.where(deg[..., None] == 0, x, lmbda(x, True) + lmbda(m) / (deg[..., None]+1e-9))


class GraphEMALayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, residual, lmbda, T):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        #self.lmbda = nn.Parameter(torch.ones(dim_in) * lmbda) # NOTE: This could be made into a learnable parameter
        #self.lmbda = lmbda
        self.lmbda = ApplyLambda(D=dim_in)
        self.T = T
        self.model = MLP(dim_in, dim_out)
        #self.lin = pyg_nn.Linear(dim_in, dim_out)

    def forward(self, batch):
        x = batch.x
        if self.dim_in == self.dim_out:
            x_in = x.clone()
        x = self.model(x)
        #x = self.lin(x)
        if self.dim_in != self.dim_out:
            x_in = x.clone()
        x = ema(x, batch.edge_index, self.lmbda, self.T)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.residual:
            x = x_in + x  # residual connection
        batch.x = x
        return batch
