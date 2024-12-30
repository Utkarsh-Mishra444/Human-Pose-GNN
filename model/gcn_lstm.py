import torch
import torch.nn.functional as F
from torch.nn import LSTM
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN_LSTM(torch.nn.Module):
    """
    Graph Convolutional Network + LSTM architecture.
    1. GCN layers extract spatial features from skeleton graphs.
    2. LSTM layers capture temporal dynamics across frames.
    """
    def __init__(self):
        super(GCN_LSTM, self).__init__()
        self.conv1 = GCNConv(5, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)

        self.lstm1 = LSTM(input_size=256, hidden_size=128, num_layers=3, batch_first=True)
        self.lstm2 = LSTM(input_size=256, hidden_size=64,  num_layers=3, batch_first=True)

        self.num_classes = 20
        self.fc1 = torch.nn.Linear(128 + 64, 128)  # We will cat the last hidden states from both LSTMs
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, self.num_classes)

    def forward(self, data_list):
        gcn_outputs = []

        # Process each 'time-step' (Data object) in data_list
        for data_t in data_list:
            x, edge_index, batch = data_t.x, data_t.edge_index, data_t.batch

            # GCN layers
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))

            # Pooling across all nodes for each sample
            x = global_mean_pool(x, batch)

            gcn_outputs.append(x)

        # Stack across the time dimension => shape: (batch_size, seq_len, feature_size)
        lstm_input = torch.stack(gcn_outputs, dim=0)

        # LSTM 1
        lstm_output_1, _ = self.lstm1(lstm_input)
        # LSTM 2
        lstm_output_2, _ = self.lstm2(lstm_input)

        # Take the last hidden state from each LSTM
        last_hidden1 = lstm_output_1[:, -1, :]
        last_hidden2 = lstm_output_2[:, -1, :]

        # Concatenate
        x = torch.cat((last_hidden1, last_hidden2), dim=1)

        # Fully-connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
