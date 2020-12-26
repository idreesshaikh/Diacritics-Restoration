import codecs
import torch


# Get characters all the possible characters

class DataSet:
    def __int__(self, filename, all_characters):
        self.filename = filename
        self.all_characters = all_characters
        self.n_letter = len(all_characters)

    # Read the large train file
    def read(self):
        alphabet = dict()
        for x in self.all_characters:
            alphabet[x] = self.all_characters.index(x)
        dataset = []
        lines = codecs.open(self.filename, encoding='utf-8').read().strip().split('\n')
        for l in lines:
            non_diacritized, diacritized = l.split('\t')
            non_diacritized = [alphabet.get(x, alphabet['*']) for x in non_diacritized]
            diacritized = [alphabet.get(x, alphabet['*']) for x in diacritized]
            dataset.append([torch.LongTensor(non_diacritized),torch.LongTensor(diacritized)])
        return dataset
