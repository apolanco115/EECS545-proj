import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class TextLSTM(nn.Module):

    def __init__(self, word_embeddings, vocab_size, params): # batch_size, output_size, hidden_size, vocab_size, embedding_length, weights):
        super(TextLSTM, self).__init__()

        self.params = params
        self.batch_size = self.params['batch_size']
        self.output_size = self.params['num_classes']
        self.hidden_size = self.params['hidden_size']
        self.embedding_length = self.params['emb_dim']
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, self.params['emb_dim'])
        self.embedding.weight = nn.Parameter(word_embeddings, requires_grad=False)

        self.dropout = self.params['dropout']
        
        self.lstm = nn.LSTM(self.embedding_length, self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.act = nn.Sigmoid()
        
        
    def forward(self, input_sentence):
        """ 
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)
        
        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)
        
        """
        
        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input = self.embedding(input_sentence) # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # input = input #.permute(1, 0, 2) # input.size() = (num_sequences, batch_size, embedding_length)
        # print(input.size())
        # if batch_size is None:
        #    h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial hidden state of the LSTM
        #    c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size)) # Initial cell state of the LSTM
        #else:
        #    h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        #    c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size))
        cur_batch_size = input.size()[1]
        input = input.permute(1, 0, 2)
        # lens = torch.tensor([embeds.size()[0],embeds.size()[1],embeds.size()[2]], device='cpu')
        h_0 = Variable(torch.zeros(1, cur_batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(1, cur_batch_size, self.hidden_size))
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, lens)
        # packed_outputs, (hidden,cell) = self.lstm(input)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # dense_outputs = self.fc(hidden)
        #outputs=self.act(dense_outputs)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        # final_hidden_state = self.dropout(final_hidden_state)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        final_output = self.fc(final_hidden_state[-1]) # final_hidden_state.size() = (1, batch_size, hidden_size) & final_output.size() = (batch_size, output_size)
        # final_output = self.sigmoid(final_output)

        return final_output

