from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import GloVe, vocab
from torch.utils.data.dataset import random_split


import torch


class Dataset:
    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')
        self.vectors = GloVe()
        self.vocab = vocab(self.vectors.stoi)
        self.vocab_size = len(self.vocab)
        self.set_specials()
        self.set_embeddings()
        self.set_data()

    def set_specials(self):
        unk_token = '<unk>'
        self.unk_index = 0

        pad_token = '<pad>'
        self.pad_index = 1

        bos_token = '<bos>'
        self.bos_index = 2
        eos_token = '<eos>'
        self.eos_index = 3

        self.vocab.insert_token(unk_token, self.unk_index)
        self.vocab.insert_token(pad_token, self.pad_index)
        self.vocab.insert_token(bos_token, self.bos_index)
        self.vocab.insert_token(eos_token, self.eos_index)

        self.vocab.set_default_index(self.unk_index)

    def set_embeddings(self):
        self.pretrained_embeds = self.vectors.vectors
        self.pretrained_embeds = torch.cat(
            (torch.zeros(1, self.pretrained_embeds.shape[1]), self.pretrained_embeds))

    def split_data(self):
        train_iter, test_iter = AG_NEWS()
        train_dataset = to_map_style_dataset(train_iter)
        test_dataset = to_map_style_dataset(test_iter)
        num_train = int(len(train_dataset) * 0.95)
        split_train, split_valid = \
            random_split(train_dataset, [
                         num_train, len(train_dataset) - num_train])
        return split_train, split_valid, test_dataset

    def process_data(self, data_itr):

        data = []
        for label, text in data_itr:
            data_tensor = torch.tensor(
                [self.vocab[token] for token in self.tokenizer(text)], dtype=torch.long)
            data.append((label-1, data_tensor))

        return data

    def set_data(self):
        train, valid, test = self.split_data()

        self.train_data = self.process_data(train)
        self.valid_data = self.process_data(valid)
        self.test_data = self.process_data(test)
