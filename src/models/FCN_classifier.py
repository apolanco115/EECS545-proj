import torch
import torch.nn as nn
import torch.nn.functional as F


class TextFCN(nn.Module):
    def __init__(self, word_embeddings, vocab_size, params):
        super(TextFCN, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(vocab_size, self.params['emb_dim'])
        self.embedding.weight = nn.Parameter(
            word_embeddings, requires_grad=False)

        self.fc1 = nn.Linear(self.params['emb_dim'], self.params['num_channels'])
        self.fc=nn.Linear(self.params['num_channels'], self.params['num_classes'] )


    def forward(self, input):
        #embedded = self.embedding(input).permute(1, 2, 0)
        embedded = self.embedding(input)
        layer1=self.fc1(embedded)
        output = self.fc(layer1)
        return output

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, loss_fn):
        self.loss_fn = loss_fn
