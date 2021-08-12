import os
import pandas as pd
from libs.nlp.lstm.utils import new_tag_list
import torch

path = os.path.abspath('..')


def load_data(type):
    print(path)
    if type == "train":
        df = pd.read_csv(f"/home/xiao.zhang/workspace/nlpcode/data/train_data/train_data.csv")
    else:
        df = pd.read_csv(f"/home/xiao.zhang/workspace/nlpcode/data/test_data/dev_data.csv")
    return df['id'], df['text'], df['label']


def load_vocab(path):
    with open(path, 'r') as f:
        vocab = f.readlines()
    word_list = [v.strip('\n') for v in vocab]
    vocab = {}
    for ind, word in enumerate(word_list):
        vocab[word] = ind
    return vocab


def process_x(x, word2ind, max_len):
    result = []
    for line in x:
        line_result = [word2ind.get(w, 1) for w in line]
        line_result += [0] * (max_len - len(line))
        result.append(line_result)
    return result


def process_y(y, tag_list, max_len):
    tag2id = {}
    for i, tag in enumerate(tag_list):
        tag2id[tag] = i

    result = []
    for line in y:
        line_list = line.split()
        line_result = [tag2id.get(t) for t in line_list]
        line_result += [0] * (max_len - len(line_list))
        result.append(line_result)
    return result


def get_dataset(type, max_len):
    try:
        id, x, y = load_data(type)
        word2ind = load_vocab("/home/xiao.zhang/workspace/nlpcode/data/vocab.txt")
        X = process_x(x, word2ind, max_len)
        Y = process_y(y, new_tag_list, max_len)
        return torch.tensor(X), torch.tensor(Y)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    x, y = get_dataset('test', 100)