import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, word_embeddings, vocab_size, params):
        super(TextCNN, self).__init__()
        self.params = params
        self.embedding = nn.Embedding(vocab_size, self.params['emb_dim'])
        self.embedding.weight = nn.Parameter(
            word_embeddings, requires_grad=False)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.params['emb_dim'], out_channels=self.params['num_channels'],
                                              kernel_size=self.params['kernel_size'][i]) for i in range(len(self.params['kernel_size']))])

        self.dropout = nn.Dropout(self.params['dropout'])

        self.fc = nn.Linear(self.params['num_channels'] *
                            (len(self.params['kernel_size'])), self.params['num_classes'])

    def forward(self, input):
        
        embedded = self.embedding(input).permute(1, 2, 0)
        conv_list = [F.relu(conv(embedded)).squeeze(2) for conv in self.convs]
        pools = [F.max_pool1d(conved, conved.shape[2]).squeeze(2)
                 for conved in conv_list]
        concat = torch.cat(pools, dim=1)
        dropped = self.dropout(concat)
        output = self.fc(dropped)
        return output
    
    def forward2(self, embedded):
        conv_list = [F.relu(conv(embedded)).squeeze(2) for conv in self.convs]
        pools = [F.max_pool1d(conved, conved.shape[2]).squeeze(2)
                 for conved in conv_list]
        concat = torch.cat(pools, dim=1)
        dropped = self.dropout(concat)
        output = self.fc(dropped)
        return output