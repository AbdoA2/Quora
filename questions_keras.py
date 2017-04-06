import pickle
import time
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, K, Lambda, Merge


f = pickle.load(open("words.pkl", 'rb'), encoding="bytes")
embeddings = f['embeddings']
f = pickle.load(open("data.pkl", 'rb'), encoding="bytes")
train1, train2, y = f['train1'][:, :40], f['train2'][:, :40], f['y']


q1 = Sequential()
q1.add(Embedding(len(embeddings), output_dim=100, weights=[embeddings], input_length=train1.shape[1], trainable=False))
q1.add(TimeDistributed(Dense(100, activation='relu')))
q1.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(100, )))

q2 = Sequential()
q2.add(Embedding(len(embeddings), output_dim=100, weights=[embeddings], input_length=train1.shape[1], trainable=False))
q2.add(TimeDistributed(Dense(100, activation='relu')))
q2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(100, )))

model = Sequential()
model.add(Merge([q1, q2], mode='concat'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

start = time.time()

model.fit([train1[:-12800, :], train2[:-12800, :]], y[:-12800, :],
          validation_data=([train1[:-12800, :], train2[:-12800, :]], y[:-12800:]),
          epochs=5, batch_size=64, verbose=1)

model.save("model.h5")
# Final evaluation of the model
scores = model.evaluate([train1[:-12800, :], train2[:-12800, :]], y[-12800:], verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

print("Total Time %s seconds" % (time.time() - start))