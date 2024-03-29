import torch
import torch.nn as nn

# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------------------
# ------------------------------NEURAL NETWORK CLASS-----------------------------

# Bidirectional LSTM (Bi Recurrent Neural Network)

class BiRNN(nn.Module):
    # initializing RNN
    def __init__(self, input_size, embed_dim, hidden_size, num_layers):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding of the all_characters/ Embedding vector is gonna b learnt by RNN
        self.embed = nn.Embedding(input_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.fully_connected = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        # We have 2 layers one going forward and one going backwards and they will be concatenated to get the hidden
        # state It's the hidden_state for that particular time_sequence.
        # hidden_state0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        # hidden_state,mini_batches,hidden_size
        # cell_state0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        batch_size = len(x)
        out = self.embed(x)
        out, (h, c) = self.lstm(out.unsqueeze(1),self.init_hidden(batch_size))
        h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        out = self.fully_connected(h)

        # out, _ = self.lstm(x,(hidden_state0,cell_state0)) #tuple out = self.fully_connected(out[:-1,:]) # take the
        # last hidden_state and all features, and send in to the linear layer
        return out

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return hidden, cell
