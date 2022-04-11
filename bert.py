import torch
import torch.nn as nn
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, Dataset
from torchtext.data.functional import to_map_style_dataset
from torchtext.datasets import AG_NEWS

from transformers import BertTokenizer, BertModel

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# /usr/venv/545bert/bin/activate
class EncodedDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, device):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index][1]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long).to(self.device, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long).to(self.device, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long).to(self.device, dtype=torch.long),
            'targets': torch.tensor(self.data[index][0], dtype=torch.long).to(self.device, dtype=torch.long)
        }




class Classifier_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 10) # 768 x 10
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 4)
        self.softmax = nn.Softmax(1)

    def forward(self, inbatch):
        output = self.bert(inbatch['ids'], attention_mask=inbatch['mask'], token_type_ids=inbatch['token_type_ids'])
        output = output.pooler_output
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)

        return output



def train_model(model, training_loader, validation_loader, optimizer, loss_func, num_epochs, device):
    best_val_loss = 1e7
    batch_size = 32
    history = {
        'loss': [],
        'accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }
    # train_losses = []
    # valid_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # running_loss = 0.0
        # running_acc = 0.0
        # history['loss'].append(0.0)
        # history['accuracy'].append(0.0)
        # history['validation_loss'].append(0.0)
        # history['validation_accuracy'].append(0.0)

        model.train()
        print(f"############# Epoch {epoch}: Training Start   #############")
        for batch_idx, data in enumerate(training_loader):

            outputs = model(data)
            #_, preds = torch.max(outputs)

            optimizer.zero_grad()
            labels = data['targets'] - torch.ones(data['targets'].shape, dtype=torch.long, device=device)
            #onehot_labels = torch.zeros((labels.shape[0], 4), dtype=torch.float, device=device)
            #d1idxs = torch.arange(start=0, end=labels.shape[0], dtype=torch.long, device=device)
            #onehot_labels[d1idxs, labels] = 1.0
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()
            # print('before loss data in training', loss.item(), train_loss)

            # train_loss += loss.item()

            outputs = torch.argmax(outputs, dim=1) + torch.ones(outputs.shape[0], device=device)
            # train_acc += (outputs == labels).float().sum() / batch_size

            train_loss += loss.item() * labels.size(0)       
            train_acc += torch.sum(outputs == labels.data)

            if batch_idx%500==0:
              batch_loss = train_loss / ((batch_idx+1) * batch_size)
              batch_acc = train_acc.double() / ((batch_idx+1) * batch_size)
              print(f'Epoch: {epoch}, Batch: {batch_idx}, Training Loss:  {batch_loss}, acc: {batch_acc}')

        history['loss'].append(train_loss/(len(training_loader) * batch_size))
        history['accuracy'].append(train_acc/(len(training_loader) * batch_size))


        print(f"############# Epoch {epoch}: Training End     #############")

        print(f'############# Epoch {epoch}: Validation Start   #############')
        ######################
        # validate the model #
        ######################

        model.eval()

        with torch.no_grad():
            for batch_idx, data in enumerate(validation_loader, 0):

                outputs = model(data)

                labels = data['targets'] - torch.ones(data['targets'].shape, dtype=torch.long, device=device)
                #onehot_labels = torch.zeros((labels.shape[0], 4), dtype=torch.float, device=device)
                #d1idxs = torch.arange(start=0, end=labels.shape[0], dtype=torch.long, device=device)
                #onehot_labels[d1idxs, labels] = 1.0

                loss = loss_func(outputs, labels)

                # val_loss += loss.item()

                outputs = torch.argmax(outputs, dim=1) + torch.ones(outputs.shape[0], device=device)

                val_loss += loss.item() * labels.size(0)
                val_acc += torch.sum(outputs == labels)

                #acc = (outputs == labels).float().sum() / batch_size
                #val_acc += acc

            history['validation_loss'].append(val_loss/(len(validation_loader) * batch_size))
            history['validation_accuracy'].append(val_acc/(len(validation_loader) * batch_size))

            print('############# Epoch {}: Validation End     #############'.format(epoch))
            # calculate average losses
            # print('before cal avg train loss', train_loss)
            # print training/validation statistics
            print(f"validation loss: {history['validation_loss'][-1]}, validation accuracy: {history['validation_accuracy'][-1]}")


            #print(f'Epoch: {epoch} \n\tAvgerage Training Loss: {train_loss/len(validation_loader)} \n\tAvgerage Training Acc: {train_acc/len(validation_loader)} \n\tAverage Validation Loss: {val_loss/len(validation_loader)} \n\tAvgerage Validation Acc: {val_loss/len(validation_loader)}')


            # ## TODO: save the model if validation loss has decreased
            if val_loss <= best_val_loss:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_val_loss, val_loss))
                # save checkpoint as best model
                torch.save(model, "./545bert-2.0")
                #save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                best_val_loss = val_loss

        print('############# Epoch {}  Done   #############\n'.format(epoch))
    print("Training loss:")
    print(history['loss'])
    print("Training accuracy:")
    print(history['accuracy'])
    print("Validation loss:")
    print(history['validation_loss'])
    print("Validation accuracy:")
    print(history['validation_accuracy'])

    return model, history




if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    print(f"Running on cuda? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    num_epochs = 64
    max_len = 64
    vsplit = 0.05
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_iter = AG_NEWS(split='train')
    trainset = to_map_style_dataset(train_iter)
    n_train = int(len(trainset) * vsplit)
    n_val = len(trainset) - n_train
    valset, trainset = random_split(trainset, [n_train, n_val])

    valset = EncodedDataset(valset, tokenizer, max_len, device)
    trainset = EncodedDataset(trainset, tokenizer, max_len, device)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    test_iter = AG_NEWS(split='test')
    testset = to_map_style_dataset(test_iter)
    testset = EncodedDataset(testset, tokenizer, max_len, device)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    print(f"Datasets loaded - train: {len(trainset)}, val: {len(valset)}, test: {len(testset)}")

    classifier_model = Classifier_model()
    if torch.cuda.is_available():
        classifier_model.cuda()

    # freeze all params except FC layer
    for name, params in classifier_model.named_parameters():
        if 'fc' not in name:
            params.requires_grad = False

    print("Model set up")

    criterion = nn.CrossEntropyLoss()



    init_lr = 1e-5
    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=init_lr)

    classifier_model, history = train_model(classifier_model, trainloader, valloader, optimizer, criterion, num_epochs, device)



    with open('outfile.txt', 'w+') as outfile:
        outfile.write(f"train loss: {str(history['loss'])}")
        outfile.write(f"train acc: {str(history['accuracy'])}")
        outfile.write(f"val loss: {str(history['validation_loss'])}")
        outfile.write(f"val acc: {str(history['validation_accuracy'])}")
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['validation_loss'], label='val loss')
    plt.savefig('./trainingplot.jpeg')
