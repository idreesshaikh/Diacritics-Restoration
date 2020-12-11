import torch
import torch.nn as nn
import string
from torch.utils.tensorboard import SummaryWriter  # print to tensorboard
from tqdm import tqdm
from BiRNN import BiRNN
from getDataSet import getDataSet

# -------------------------------------------------------------------------------

# Deice Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ----------------------------CHECKPOINT----------------------------------

def save_checkpoint(state, filename="my_saved_checkpoint.pth.tar"):
    print("Saving your checkpoint ehh: you so lazy")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# ================================================================================


# ================================================================================

# --------------------------------------------------------------------------------
def main():
    all_characters = string.printable + 'áéíóöőúüűÁÉÍÓÖŐÚÜŰ'
    n_characters = len(all_characters)

    # -------------------------Hyperparameters----------------------------------------

    input_size = n_characters
    num_layers = 2
    embed_dim = 10
    hidden_size = 10
    learning_rate = 0.001
    num_epochs = 2
    load_model = True

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # GETTING THE DATA from DATASET

    train_data = getDataSet()
    train_data.__int__('diacritic_data/train', all_characters)
    train_loader = train_data.read()
    valid_data = getDataSet()
    valid_data.__int__('diacritic_data/dev', all_characters)
    valid_loader = valid_data.read()
    test_data = getDataSet()
    test_data.__int__('diacritic_data/test', all_characters)
    test_loader = test_data.read()
    #            'Diacritics': [line for line in with_diacritics[1:1000]]}
    # --------------------------------------------------------------------------------

    # Initialize Network
    model = BiRNN(input_size=input_size, embed_dim=embed_dim, hidden_size=hidden_size, num_layers=num_layers).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    Writer = SummaryWriter(f'runs/lines0')  # for tensorboard

    # --------------------------------------------------------------------------------

    # Loading my saved checkedpoint
    if load_model:
        load_checkpoint((torch.load("my_saved_checkpoint.pth.tar")), model, optimizer)

    # --------------------------------------------------------------------------------

    # Training Network
    print("Start training")
    for epoch in range(num_epochs):
        losses = []

        if epoch % 2 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, (data, targets) in loop:
            # Get data to cuda if possible
            data = data.to(device=device)  # Remove the one for a particular axis
            targets = targets.to(device=device)

            #data = data.reshape(data.shape[0], )

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradiant descent or adam step
            optimizer.step()

            Writer.add_scalar('Training Loss', loss, global_step=epoch)

            # update progress bar
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item(), acc=torch.rand(1).item())

    # --------------------------------------------------------------------------------

    # Check Accuracy
    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0
        model.eval()

        with torch.no_grad():
            # matches, total = 0, 0
            for x, y in loader:
                x = x.to(device=device)
                y = y.to(device=device)
                #x = x.reshape(x.shape[0], -1)
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                # Calculate accuracy
            accuracy = (float(num_correct) / float(num_samples)) * 100
            print(f'Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}')
        model.train()
        return accuracy

    check_accuracy(train_loader, model)
    check_accuracy(valid_loader, model)
    check_accuracy(test_loader, model)


if __name__ == '__main__':
    main()
