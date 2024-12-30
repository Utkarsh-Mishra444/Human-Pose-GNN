import torch
from torch_geometric.data import Data
from .edge_index import get_edge_index

def prepare_data(X_train, y_train, X_val, y_val):
    """
    Creates lists of torch_geometric.data.Data objects for training and validation.
    """
    edge_index = get_edge_index()

    train_data = [
        Data(
            x=torch.tensor(X_train[i], dtype=torch.float32),
            y=torch.tensor(y_train[i], dtype=torch.long),
            edge_index=edge_index
        )
        for i in range(len(X_train))
    ]

    val_data = [
        Data(
            x=torch.tensor(X_val[i], dtype=torch.float32),
            y=torch.tensor(y_val[i], dtype=torch.long),
            edge_index=edge_index
        )
        for i in range(len(X_val))
    ]

    return train_data, val_data
