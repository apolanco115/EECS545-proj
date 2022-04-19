import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.LSTM_classifier import TextLSTM
from data.preprocess import Dataset
import numpy as np
import matplotlib.pyplot as plt


def main():

    params = {
        'emb_dim':  300,
        'num_classes': 4,
        'epochs': 30,
        'lr': 0.0001,
        'batch_size': 64,
        'dropout': 0.05,
        'hidden_size': 16,
        'hidden_layers': 2,
        'is_bidirectional': True
    }

    ag_news = Dataset()
    model = TextLSTM(word_embeddings=ag_news.pretrained_embeds,
                    vocab_size=ag_news.vocab_size, params=params)
    epoch_avg_train_loss = np.zeros((params['epochs'],))
    epoch_avg_valid_loss = np.zeros((params['epochs'],))
    cur_epoch = np.ones((1,), dtype=np.int64)

    ## TO LOAD OLD MODEL & DATA
    #model.load_state_dict(torch.load('TextLSTM_state.pt'))
    #model.eval()
    #epoch_avg_train_loss = torch.load('LSTM_epoch_avg_train_loss.pt')
    #epoch_avg_valid_loss = torch.load('LSTM_epoch_avg_valid_loss.pt')
    #cur_epoch = torch.load('cur_epoch.pt')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

    def generate_batch(data_batch):
        label_batch, text_batch = [], []
        for (l_item, t_item) in data_batch:
            label_batch.append(l_item)
            text_batch.append(torch.cat(
                [torch.tensor([ag_news.bos_index]), t_item, torch.tensor([ag_news.eos_index])], dim=0))
        text_batch = pad_sequence(text_batch, padding_value=ag_news.pad_index)
        label_batch = torch.tensor(label_batch, dtype=torch.int64)
        return label_batch, text_batch

    train_loader = DataLoader(ag_news.train_data, batch_size=params['batch_size'],
                              shuffle=True, collate_fn=generate_batch)
    valid_loader = DataLoader(ag_news.valid_data, batch_size=params['batch_size'],
                              shuffle=True, collate_fn=generate_batch)
    test_loader = DataLoader(ag_news.test_data, batch_size=params['batch_size'],
                             shuffle=True, collate_fn=generate_batch)

    print('begin training')
    for epoch in range(cur_epoch[0], params['epochs'] + 1):
        epoch_start_time = time.time()
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 100
        start_time = time.time()

        print('-' * 69)
        print('| begin epoch {:3d} |'.format(epoch))
        print('-' * 69)

        epoch_train_loss = 0.0
        epoch_train_count = 0.0

        for idx, (label, text) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

            epoch_train_loss += loss
            epoch_train_count += 1

            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(train_loader),
                                                                 total_acc/total_count, loss))
                total_acc, total_count = 0, 0
                start_time = time.time()

        model.eval()
        total_acc, total_count = 0, 0

        epoch_avg_train_loss[epoch-1] = (epoch_train_loss / epoch_train_count)

        epoch_valid_loss = 0.0
        epoch_valid_count = 0.0

        with torch.no_grad():
            for idx, (label, text) in enumerate(valid_loader):
                predicted_label = F.softmax(model(text), dim=1)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
                epoch_valid_loss += loss 
                epoch_valid_count += 1
        accu_val = total_acc/total_count

        epoch_avg_valid_loss[epoch-1] = (epoch_valid_loss / epoch_valid_count)

        cur_epoch[0] = epoch + 1

        print('Saving...')
        ## SAVE MODEL STATE AND LOSS HISTORY
        torch.save(model.state_dict(), 'TextLSTM_state_epoch30_hidden16.pt')
        np.save('LSTM_epoch_avg_train_loss_e30_h16', epoch_avg_train_loss)
        np.save('LSTM_epoch_avg_valid_loss_e30_h16', epoch_avg_valid_loss)
        np.save('cur_epoch_e30_e16', cur_epoch)
        print('Save complete')

        print('-' * 69)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 69)
        


    print('Training Complete!')
    print('Checking the results of test dataset.')
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in enumerate(test_loader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    accu_test = total_acc/total_count
    print('test accuracy {:8.3f}'.format(accu_test))


    ## PLOT TRAIN/VALID LOSS OVER TIME
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(epoch_avg_train_loss)
    ax.plot(epoch_avg_valid_loss)
    # ax.set_title('LSTM Classifier')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(['training loss','validation loss'])
    fig.savefig('LSTM_loss_graph_epoch30_hidden16.png')


if __name__ == '__main__':
    main()

