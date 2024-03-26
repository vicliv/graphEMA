import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn



class GraphEMALayer(nn.Module):
    """
    GraphEMA layer that calculates message passing usng exponential moving average.
    """
    def __init__(self, dim_in, dim_out, dropout, residual):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual

        self.model = nn.Identity() #EMAMessagePassing()
        self.mlp = nn.Sequential(
            torch.nn.Linear(dim_in, dim_in),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_in, dim_out))
        # MLP to transform node embeddings

    def forward(self, batch):
        x_in = batch.x

        #batch.x = self.model(batch.x, batch.x0, batch.edge_index)
        batch.x = self.model(batch.x) #Curreny for debugging

        batch.x = self.mlp(batch.x)
        batch.x = F.relu(batch.x)
        batch.x = F.dropout(batch.x, p=self.dropout, training=self.training)
        breakpoint()
        if self.residual:
            batch.x = x_in + batch.x  # residual connection

        return batch