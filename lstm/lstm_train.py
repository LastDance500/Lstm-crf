import torch
import torch.utils.data as data
import torch.optim as optim
from libs.nlp.lstm.utils.utils import get_dataset
from libs.nlp.lstm.utils import config
from libs.nlp.lstm.utils import metrics
from libs.nlp.lstm.src.lstm_model import Bi_LSTM_CRF
from libs.nlp.lstm.utils import new_tag_list, id2tag


class Dataset(data.Dataset):
    def __init__(self, type, config):
        self.x, self.y = get_dataset(type, config['max_len'])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, indx):
        return self.x[indx], self.y[indx]


def train(model, config, device):
    try:
        model.to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        data_loader = data.DataLoader(Dataset('train', config), batch_size=config['batch_size'])
        best_acc = 0
        for epoch in range(config['epochs']):
            total_loss, total_acc = 0, 0
            for batch, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                model.zero_grad()
                loss = model.score(x, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = model.predict(x)
                pred = torch.Tensor(pred).to(device)
                sequence_lengths = torch.sum(torch.ne(x, 0), dim=1)
                acc = metrics(m='recall', y=y, pred=pred, sequence_lengths=sequence_lengths)
                total_acc += acc.item()
                if (batch + 1) % 5 == 0:
                    print('[Epoch{} Batch{}] loss:{:.3f} acc:{:.3f}'.format(epoch + 1, batch + 1, loss.data, acc))
            print('Epoch{} Loss: {:.5f} Accuracy: {:.5f}'.format(epoch + 1, total_loss / len(data_loader),
                                                                 total_acc / len(data_loader)))
            test_acc = valid('recall', model, 200, device)
            print('_________')
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'model/jd_ner_model.bin')
                print('saving model with acc {:.3f}'.format(test_acc))
            model.train()
    except Exception as e:
        print(e)


def valid(m, model, batch_size, device):
    model.eval()
    data_loader = data.DataLoader(Dataset('test', config), batch_size=batch_size)
    total_acc = 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model.predict(x)
            pred = torch.Tensor(pred).to(device)
            sequence_lengths = torch.sum(torch.ne(x, 0), dim=1)
            acc = metrics(m=m, y=y, pred=pred, sequence_lengths=sequence_lengths)
            total_acc += acc
        print('On test set:')
        print('Accuracy: {:.5f}'.format(total_acc/len(data_loader)))

    return total_acc/len(data_loader)


def predict(model, device):
    model.load_state_dict(torch.load('model/jd_ner_model.bin'))
    model.to(device)
    data_loader = data.DataLoader(Dataset('test', config), batch_size=1)
    with torch.no_grad():
        for batch, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model.predict(x)
            pred = torch.Tensor(pred).to(device)
            Y = [id2tag[str(int(id.item()))] for id in y[0]]
            P = [id2tag[str(int(id.item()))] for id in pred[0]]
            print("预测:", P)
            print("标注:", Y)
            print("\n")
            if batch == 10:
                break


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Bi_LSTM_CRF(config['vocab_size'], config['embedding_dim'], config['model_dim'], len(new_tag_list))
    # model = torch.load('../models/model.pkl', map_location=torch.device(device))

    train(model, config, device)
    # valid('recall',model, 200, device)
    predict(model, device)
