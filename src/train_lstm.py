import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.LSTM_classifier import TextLSTM
from data.preprocess import Dataset


def main():

    params = {
        'emb_dim':  300,
        'num_classes': 4,
        'epochs': 15,
        'lr': 0.0001,
        'batch_size': 32,
        'dropout': 0.05,
        'hidden_size': 256
    }

    ag_news = Dataset()
    model = TextLSTM(word_embeddings=ag_news.pretrained_embeds,
                    vocab_size=ag_news.vocab_size, params=params)

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
    for epoch in range(1, params['epochs'] + 1):
        epoch_start_time = time.time()
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 100
        start_time = time.time()

        for idx, (label, text) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f} | loss {:8.3f}'.format(epoch, idx, len(train_loader),
                                                                 total_acc/total_count, loss))
                total_acc, total_count = 0, 0
                start_time = time.time()

        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text) in enumerate(valid_loader):
                predicted_label = F.softmax(model(text), dim=1)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        accu_val = total_acc/total_count

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


if __name__ == '__main__':
    main()

