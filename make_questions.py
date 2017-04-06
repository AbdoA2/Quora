import numpy as np
import pickle
import csv
import re


def words2ints(text, word2int):
    text = re.findall(r"[a-z]+|[.,!?;:_0-9+\-\\/*)(]", text)
    numbers = []
    for word in text:
        if word in word2int:
            numbers.append(word2int[word])
        elif word.endswith("'s") and word[:-2] in word2int:
            numbers.append(word2int[word[:-2]])
            numbers.append(word2int["'s"])
        elif word.endswith("n't") and word[:-3] in word2int:
            numbers.append(word2int[word[:-3]])
            numbers.append(word2int["n't"])
        elif word.endswith("'m") and word[:-2] in word2int:
            numbers.append(word2int[word[:-2]])
            numbers.append(word2int["'m"])
        elif word.startswith("'") and word.endswith("'") and word[1:-1] in word2int:
            numbers.append(word2int["'"])
            numbers.append(word2int[word[1:-1]])
            numbers.append(word2int["'"])
        elif word.startswith("'") and word[1:] in word2int:
            numbers.append(word2int["'"])
            numbers.append(word2int[word[1:]])
        elif word.endswith("'") and word[:-1] in word2int:
            numbers.append(word2int[word[:-1]])
            numbers.append(word2int["'"])
    return numbers


def get_data(filename, word2int, col1, col2, labels=False, batches=0):
    train1, train2, len_train1, len_train2, y_train = [], [], [], [], []
    max_len = 0
    f = open(filename, encoding="utf8")
    reader = csv.DictReader(f)
    l1 = l2 = 0

    for line in reader:
        if labels:
            y_train.append(int(line['is_duplicate']))
        q1 = line[col1].lower()
        q2 = line[col2].lower()
        q1 = words2ints(q1, word2int)
        q2 = words2ints(q2, word2int)
        l1 += len(q1)
        l2 += len(q2)
        train1.append(q1)
        len_train1.append(len(train1[-1]))
        train2.append(q2)
        len_train2.append(len(train2[-1]))
        max_len = max(max_len, len_train1[-1], len_train2[-1])
        t = 1 - int(line['is_duplicate'])

    train1 = [i + [0] * (max_len - len(i)) for i in train1]
    train1 = np.matrix(train1, dtype=np.int32)
    len_train1 = np.array(len_train1, dtype=np.int32)
    train2 = [i + [0] * (max_len - len(i)) for i in train2]
    train2 = np.matrix(train2, dtype=np.int32)
    len_train2 = np.array(len_train2, dtype=np.int32)
    y = np.matrix(y_train, dtype=np.int32)
    return train1, len_train1, train2, len_train2, y.T

f = pickle.load(open("words.pkl", 'rb'), encoding="bytes")
word2int = f['word2int']


train1, len_train1, train2, len_train2, y = get_data("train.csv", word2int, "question1", "question2", True)
f = open("data.pkl", "wb")
data = {'train1': train1, 'train2': train2, 'len_train1': len_train1, 'len_train2': len_train2, 'y': y}
pickle.dump(data, f)
f.close()

'''
test1, len_test1, test2, len_test2, y = get_data("test.csv", word2int, "question1", "question2")
f = open("test.pkl", "wb")
data = {'test1': test1, 'test2': test2, 'len_test1': len_test1, 'len_test2': len_test2, 'y': y}
pickle.dump(data, f)
f.close()
'''