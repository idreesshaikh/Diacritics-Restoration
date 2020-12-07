import string
import torch


# Get characters all the possible characters
class getDataSet:
    def __int__(self, filename, all_characters):
        self.filename = filename
        self.all_characters = all_characters

    # Read the large train file
    def read(self):
        alphabet = dict()
        for x in self.all_characters:
            alphabet[x] = self.all_characters.index(x)
        dataset = []
        lines = open(self.filename, encoding='utf-8').read().strip().split('\n')
        for l in lines:
            non_diacritized, diacritized = l.split('\t')
            non_diacritized = [alphabet.get(x, alphabet['*']) for x in non_diacritized]
            diacritized = [alphabet.get(x, alphabet['*']) for x in diacritized]
            dataset.append([torch.tensor(non_diacritized), torch.tensor(diacritized)])
        return dataset

    """
    def char__tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor
    """
