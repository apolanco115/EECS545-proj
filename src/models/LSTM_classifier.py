import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class TextLSTM(nn.Module):

    def __init__(self, word_embeddings, vocab_size, params): # batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(TextLSTM, self).__init__()

        self.params = params
        
        self.embeddings = nn.Embedding(vocab_size, self.params['emb_dim'])
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        self.lstm = nn.LSTM(input_size=self.params['emb_dim'],
                            hidden_size=self.params['hidden_size'],
                            num_layers=self.params['hidden_layers'],
                            dropout=self.params['dropout'],
                            bidirectional=self.params['is_bidirectional'])

        self.dropout = nn.Dropout(self.params['dropout'])

        self.fc = nn.Linear(self.params['hidden_size'] * self.params['hidden_layers']* (1+self.params['is_bidirectional']),
                            self.params['num_classes'])

        
    def forward(self, x):
       embedded_sent = self.embeddings(x)

       lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
       final_feature_map = self.dropout(h_n)

       final_feature_map = torch.cat([final_feature_map[i,:,:] for i in range(final_feature_map.shape[0])], dim=1)
       final_out = self.fc(final_feature_map)

       return final_out