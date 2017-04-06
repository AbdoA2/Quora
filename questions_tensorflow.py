import numpy as np
import tensorflow as tf
import pickle
import time
from eval import evaluate


class Questions:
    def __init__(self, embeddings, max_len):
        self.embeddings = embeddings
        self.prediction = None
        self.loss = None
        self.sess = tf.Session()
        self.path = "questions_log"
        self.max_len = max_len
        self.train_step = None
        self.q1 = self.q2 = self.seq_len1 = self.seq_len2 = self.labels = self.keep_prob = None
        self.build_inputs()

    def build_inputs(self):
        self.q1 = tf.placeholder(tf.int32, [128, self.max_len])
        self.q2 = tf.placeholder(tf.int32, [128, self.max_len])
        self.seq_len1 = tf.placeholder(tf.int32, [128])
        self.seq_len2 = tf.placeholder(tf.int32, [128])
        self.labels = tf.placeholder(tf.float32, [128, 2])
        self.keep_prob = tf.placeholder(tf.float32)

    def batch_norm_wrapper(self, inputs, is_training):

        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros(inputs.get_shape()[1:]), trainable=False)
        pop_var = tf.Variable(tf.ones(inputs.get_shape()[1:]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, batch_mean)
            train_var = tf.assign(pop_var, batch_var)
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean, batch_var, beta, scale, 1e-3)
        else:
            return tf.nn.batch_normalization(inputs,
                                             pop_mean, pop_var, beta, scale, 1e-3)

    @staticmethod
    def _weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)

    def construct(self, is_training, num_neurons=100, num_layers=1):
        cell1 = tf.nn.rnn_cell.LSTMCell(num_neurons)  # Or LSTMCell(num_neurons)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=self.keep_prob)
        cell1 = tf.nn.rnn_cell.MultiRNNCell([cell1] * num_layers)

        cell2 = tf.nn.rnn_cell.LSTMCell(num_neurons)  # Or LSTMCell(num_neurons)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=self.keep_prob)
        cell2 = tf.nn.rnn_cell.MultiRNNCell([cell2] * num_layers)

        v1 = tf.nn.embedding_lookup(tf.Variable(self.embeddings), self.q1)
        v2 = tf.nn.embedding_lookup(tf.Variable(self.embeddings), self.q2)

        output1, _ = tf.nn.dynamic_rnn(cell1, v1, dtype=tf.float32, sequence_length=self.seq_len1, scope='n1')
        output2, _ = tf.nn.dynamic_rnn(cell2, v2, dtype=tf.float32, sequence_length=self.seq_len2, scope='n2')
        #output = tf.transpose(output, [1, 0, 2])

        flatten1 = tf.reshape(output1, [-1, num_neurons])
        last1 = tf.gather(flatten1, self.seq_len1)

        flatten2 = tf.reshape(output2, [-1, num_neurons])
        last2 = tf.gather(flatten2, self.seq_len2)

        diff = tf.exp(tf.abs(last1 - last2))
        w1, b1 = self._weight_and_bias(num_neurons, 2)
        self.prediction = tf.nn.softmax(tf.matmul(diff, w1) + b1)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))

    def train(self, X1, X2, y, train_len1, train_len2, learning_rate, batch_size, steps, scratch):
        self.construct(True)
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        saver = tf.train.Saver()

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            if not scratch:
                saver.restore(self.sess, self.path + "/model.ckpt")
            for i in range(steps):
                # get a new batch
                indices = np.random.choice(X1.shape[0], size=batch_size, replace=True)
                X1_batch = X1[indices]
                X2_batch = X2[indices]
                y_batch = y[indices]
                len_batch1 = train_len1[indices]
                len_batch1 -= 1
                len_batch1 += np.array(range(0, batch_size*self.max_len, self.max_len))
                len_batch2 = train_len2[indices]
                len_batch2 -= 1
                len_batch2 += np.array(range(0, batch_size * self.max_len, self.max_len))

                # measure the accuracy every 100 iterations
                if i % 100 == 0:
                    train_accuracy = self.sess.run(self.loss, feed_dict={
                        self.q1: X1_batch, self.q2: X2_batch,
                        self.seq_len1: len_batch1, self.seq_len2: len_batch2,
                        self.labels: y_batch, self.keep_prob: 1.0})
                    print("step %d, training loss %g" % (i, train_accuracy))

                # train the model
                self.train_step.run(feed_dict={
                    self.q1: X1_batch, self.q2: X2_batch,
                    self.seq_len1: len_batch1, self.seq_len2: len_batch2,
                    self.labels: y_batch, self.keep_prob: 1.0
                })

                if (i + 1) % 100 == 0:
                    saver.save(self.sess, self.path + "/model.ckpt")

    def predict(self, X1, X2, x1_len, x2_len, batch_size):
        # Remove any previous varaibles from the training
        tf.reset_default_graph()
        self.sess.close()

        # Build the model again
        self.build_inputs()
        self.sess = tf.Session()
        self.construct(False)

        labels = []
        with self.sess.as_default():
            # Reload the last model
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.sess, self.path + "/model.ckpt")

            # Make predictions
            labels = []
            for i in range(len(X1) // batch_size):
                len_batch1 = x1_len[i * batch_size:(i+1)*batch_size]
                len_batch1 -= 1
                len_batch1 += np.array(range(0, batch_size * self.max_len, self.max_len))

                len_batch2 = x2_len[i * batch_size:(i + 1) * batch_size]
                len_batch2 -= 1
                len_batch2 += np.array(range(0, batch_size * self.max_len, self.max_len))

                ans = self.prediction.eval(feed_dict={
                        self.q1: X1[i * batch_size:(i+1)*batch_size, :], self.seq_len1:len_batch1,
                        self.q2: X2[i * batch_size:(i + 1) * batch_size, :], self.seq_len2: len_batch2,
                        self.keep_prob: 1.0,})
                labels += [label for label in tf.argmax(ans, 1).eval()]
                print(i)

            # i = len(X1) - (len(X1) % batch_size)
            # len_batch1 = x1_len[i:]
            # len_batch1 -= 1
            # len_batch1 += np.array(range(0, (len(X1) % batch_size) * self.max_len, self.max_len))
            #
            # len_batch2 = x2_len[i:]
            # len_batch2 -= 1
            # len_batch2 += np.array(range(0, (len(X1) % batch_size) * self.max_len, self.max_len))
            # ans = self.prediction.eval(feed_dict={
            #         self.q1: X1[i:, :], self.seq_len1:len_batch1,
            #         self.q2: X2[i:, :], self.seq_len2: len_batch2,
            #         self.keep_prob: 1.0})
            # labels += [label for label in tf.argmax(ans, 1).eval()]

        return labels

f = pickle.load(open("words.pkl", 'rb'), encoding="bytes")
embeddings = f['embeddings']
f = pickle.load(open("data1.pkl", 'rb'), encoding="bytes")
train1, len_train1, train2, len_train2, y = f['train1'], f['len_train1'], f['train2'], f['len_train2'], f['y']
q_classifier = Questions(embeddings, train1.shape[1])
start = time.time()
q_classifier.train(train1[:-12800, :], train2[:-12800, :], y[:-12800], len_train1[:-12800], len_train2[:-12800], 0.001, 128, 301, False)
print("duration: %d" % (time.time() - start))
predicted = q_classifier.predict(train1[-12800:, :], train2[-12800:, :], len_train1[-12800:], len_train2[-12800:], 128)
evaluate(predicted, np.argmax(y[-12800:], axis=1), 2)