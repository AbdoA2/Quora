import numpy as np
import pickle

f = open("glove.6B.100d.txt", encoding="utf8")
embeddings = [[0]*100]
i = 1
word2int = {'not-found': 0}
int2word = []
for line in f:
    vec = line.split()
    embeddings.append(list(map(float, vec[1:])))
    word2int[vec[0]] = i
    int2word.append(vec[0])
    i += 1

embeddings = np.matrix(embeddings, dtype=np.float32)
f = open("words.pkl", "wb")
pickle.dump({'embeddings': embeddings, 'word2int': word2int, 'int2word': int2word}, f)
f.close()
