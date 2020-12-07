import torch
import torch.nn as nn


# -------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------------------------------------------------------------
# ------------------------------NEURAL NETWORK CLASS-----------------------------

# Bidirectional LSTM (Bi Recurrent Neural Network)

class BiRNN(nn.Module):
    # initializing RNN
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding of the all_characters/ Embedding vector is gonna b learnt by RNN
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=True)
        self.fully_connected = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, hidden, cell):
        # We have 2 layers one going forward and one going backwards and they will be concatenated to get the hidden
        # state It's the hidden_state for that particular time_sequence.
        #hidden_state0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)
        #hidden_state,mini_batches,hidden_size
        #cell_state0 = torch.zeros(self.num_layers*2,x.size(0),self.hidden_size).to(device)

        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fully_connected(out.reshape(out.shape[0], -1))

        # out, _ = self.lstm(x,(hidden_state0,cell_state0)) #tuple out = self.fully_connected(out[:-1,:]) # take the
        # last hidden_state and all features, and send in to the linear layer
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        return hidden, cell
