import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=1, softplus=True):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.out_act = nn.Softplus() if softplus else nn.Identity()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.linear(lstm_out)
        return self.out_act(out.squeeze())
