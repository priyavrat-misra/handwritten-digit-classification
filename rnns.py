import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)  # '10' here means num_classes
        # nn.Linear(hidden_size*28, 10)  # when all the sequences' o/p are used

    def forward(self, t):
        h_0 = torch.zeros(self.num_layers, t.size(0),
                          self.hidden_size).to(device)

        # when all the sequences' o/p are used:
        # out, _ = self.rnn(t, h_0)
        # out = out.reshape(out.size(0), -1)
        # out = self.fc(out)

        # only the features from last sequence are used,
        # as they have information from the previous sequences.
        out, _ = self.rnn(t, h_0)
        out = self.fc(out[:, -1, :])

        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, t):
        h_0 = torch.zeros(self.num_layers, t.size(0),
                          self.hidden_size).to(device)

        out, _ = self.gru(t, h_0)
        out = self.fc(out[:, -1, :])

        return out


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, t):
        h_0 = torch.zeros(self.num_layers, t.size(0),
                          self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, t.size(0),
                          self.hidden_size).to(device)

        out, _ = self.lstm(t, (h_0, c_0))
        out = self.fc(out[:, -1, :])

        return out


class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, 10)

    def forward(self, t):
        h_0 = torch.zeros(self.num_layers*2, t.size(0),
                          self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers*2, t.size(0),
                          self.hidden_size).to(device)

        out, _ = self.lstm(t, (h_0, c_0))
        out = self.fc(out[:, -1, :])

        return out
