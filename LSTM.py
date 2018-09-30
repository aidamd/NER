import numpy as np
import tensorflow as tf
import math
import sys
sys.path.insert(1, '..')
from tensorflow.contrib.layers import fully_connected
from tensorflow.python.ops import array_ops
from utils import conlleval
from utils import data_converter


class LSTM():
    def __init__(self, params, vocabs, embeddings, n_outputs, chars, char_embeddings):
        self.params = params
        self.vocab = vocabs
        self.chars = chars
        self.n_outputs = n_outputs
        for key in params:
            setattr(self, key, params[key])
        if self.pretrain:
            self.my_embeddings = embeddings
        self.char_embeddings = char_embeddings


    def build_embedding(self):
        if self.pretrain:
            embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.vocab), 300]),
                                     trainable=False)

            embedding_placeholder = tf.placeholder(tf.float32, [len(self.vocab), 300])
            embedding_init = embeddings.assign(embedding_placeholder)
        else:
            embedding_placeholder = tf.get_variable("embedding",
                                                    initializer=tf.random_uniform(
                                                        [len(self.vocab), 300], -1, 1),
                                                    dtype=tf.float32)
        return embedding_placeholder

    def build_char_embedding(self):
        #char_embedding_placeholder = tf.get_variable("char_embedding", dtype=tf.float32, shape=[len(self.chars) + 2, 100])
        char_embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.chars), 300]),
                                 trainable=False)

        char_embedding_placeholder = tf.placeholder(tf.float32, [len(self.chars), 300])
        char_embedding_init = char_embeddings.assign(char_embedding_placeholder)
        return char_embedding_placeholder

    def build_feature_embedding(self):
        # char_embedding_placeholder = tf.get_variable("char_embedding", dtype=tf.float32, shape=[len(self.chars) + 2, 100])
        char_embeddings = tf.Variable(tf.constant(0.0, shape=[len(self.chars), 300]),
                                      trainable=False)

        char_embedding_placeholder = tf.placeholder(tf.float32, [len(self.chars), 300])
        char_embedding_init = char_embeddings.assign(char_embedding_placeholder)
        return char_embedding_placeholder


    def drop_padding(self, array, length):
        relevant = tf.gather(array, length - tf.ones_like(length), axis = 1)
        #relevant_flat = tf.reshape(relevant, [-1])
        return relevant


    def build(self):
        # embedding placeholder
        self.embedding_placeholder = self.build_embedding()
        self.max_data = tf.placeholder(tf.int32)
        self.inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")

        
        # inputs are converted to their embedding representations
        self.embed = tf.nn.embedding_lookup(self.embedding_placeholder, self.inputs)

        # list of sentence lengths
        self.sequence_length = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

        if self.cell == "LSTM":
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_size)
        elif self.cell == "GRU":
            cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

        # a dropout cell is present after the nn cell
        cell_drop = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.keep_prob)
        network = tf.contrib.rnn.MultiRNNCell([cell_drop] * self.num_layers)

        self.y = tf.placeholder(tf.int64, [None, None])

        # the labels without paddings
        self.y_dropped = self.drop_padding(self.y, self.sequence_length)

        # appends the outputs of character biLSTM to the word embeddings
        if self.char:
            embed = self.get_char_layer()
        else:
            embed = self.embed

        # adds a dropout layer on the embeddings
        embed = tf.layers.dropout(embed, rate=self.keep_prob)

        if self.model == "BiLSTM":
            rnn_outputs, states = tf.nn.bidirectional_dynamic_rnn(network, network, embed,
                                                                  dtype=tf.float32, sequence_length=self.sequence_length)
            fw_outputs, bw_outputs = rnn_outputs
            self.outputs = tf.concat([fw_outputs, bw_outputs], 2)
        else:
            self.rnn_outputs, self.states = tf.nn.dynamic_rnn(network, embed,
                                                              dtype=tf.float32, sequence_length=self.sequence_length)
            self.outputs = self.rnn_outputs

        if self.features:
            self.feat_inputs = tf.placeholder(tf.float32, shape=[None, None, self.feat_len])
            self.outputs = tf.concat([self.outputs, self.feat_inputs], axis=-1)

        self.logits = fully_connected(tf.layers.dropout(self.outputs, rate=self.keep_prob), self.hidden_size)
        self.logits = fully_connected(tf.layers.dropout(self.logits, rate=self.keep_prob), self.n_outputs,)

        if self.crf:
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.y, self.sequence_length)
            self.loss = tf.reduce_mean(-log_likelihood)
            predicted_label, viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.transition_params, self.sequence_length)
            self.predicted_label = tf.cast(predicted_label, tf.int64)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
            mask = tf.sequence_mask(self.sequence_length)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)
            self.predicted_label = self.drop_padding(tf.argmax(self.logits, 2), self.sequence_length)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted_label, self.y), tf.float32))

        self.training_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def run_model(self, batches, test_batches, char_batches, test_char_batches, test_sents, tags, train_feat, test_feat, testb_feat, testb_char_batch, testb_batch, testb, max_test):
        init = tf.global_variables_initializer()

        with tf.Session() as self.sess:
            init.run()
            epoch = 1
            while True:
                ## Train
                epoch_loss = float(0)
                acc_train = 0
                epoch += 1
                for batch in range(len(batches)):
                    X_batch, X_len, y_batch = batches[batch]
                    feed_dict = {self.inputs: X_batch,
                                                      self.sequence_length: X_len,
                                                      self.keep_prob: self.keep_ratio,
                                                      self.y:y_batch,
                                                      self.max_data: self.max_len
                                }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    if self.char:
                        X_char_batch, X_char_len = char_batches[batch]
                        feed_dict[self.char_embed_placeholder]= self.char_embeddings
                        feed_dict[self.char_inputs] = X_char_batch
                        feed_dict[self.word_length] = X_char_len
                    if self.features:
                        feed_dict[self.feat_inputs] = train_feat[batch]
                    _, loss_val, pred = self.sess.run([self.training_op, self.loss, self.predicted_label], feed_dict=feed_dict)
                    #print(pred)
                    acc_train += self.accuracy.eval(feed_dict=feed_dict)
                    epoch_loss += loss_val

                ## Test
                acc_test = 0
                pred = list()
                for batch in range(len(test_batches)):
                    X_batch, X_len, y_batch = test_batches[batch]
                    feed_dict = {self.inputs: X_batch, self.sequence_length: X_len,
                                                        self.keep_prob: 1,
                                                        self.y:y_batch,
                                                        self.max_data: self.max_len
                                 }
                    if self.pretrain:
                        feed_dict[self.embedding_placeholder] = self.my_embeddings
                    if self.char:
                        X_char_batch, X_char_len = test_char_batches[batch]
                        feed_dict[self.char_inputs] = X_char_batch
                        feed_dict[self.word_length] = X_char_len
                        feed_dict[self.char_embed_placeholder] = self.char_embeddings
                    if self.features:
                        feed_dict[self.feat_inputs] = test_feat[batch]

                    acc_test += self.accuracy.eval(feed_dict=feed_dict)
                    if epoch % 20 == 2:
                        pred.extend(self.predicted_label.eval(feed_dict=feed_dict).tolist())

                if epoch % 20 == 2:
                    for sent in range(len(test_sents)):
                        for word in range(len(test_sents[sent])):
                            pred[sent][word] = tags[pred[sent][word]]
                        pred[sent] = pred[sent][:len(test_sents[sent])]
                    new_conll_sents = data_converter.add_column(test_sents, data_converter.tags_to_conll(pred))
                    conlleval.evaluate(data_converter.conll_to_data_stream(new_conll_sents, write_to_file="rnn.testa"))


                print(epoch, "Train accuracy:", acc_train / float(len(batches)),
                      "Loss: ", epoch_loss / float(len(batches)),
                      "Test accuracy: ", acc_test / float(len(test_batches)))

                if epoch == 100:
                    pred = list()
                    for batch in range(len(testb_batch)):
                        X_batch, X_len, y_batch = testb_batch[batch]
                        feed_dict = {self.inputs: X_batch, self.sequence_length: X_len,
                                     self.keep_prob: 1,
                                     self.max_data: max_test
                                     }
                        if self.pretrain:
                            feed_dict[self.embedding_placeholder] = self.my_embeddings
                        if self.char:
                            X_char_batch, X_char_len = testb_char_batch[batch]
                            feed_dict[self.char_inputs] = X_char_batch
                            feed_dict[self.word_length] = X_char_len
                            feed_dict[self.char_embed_placeholder] = self.char_embeddings
                        if self.features:
                            feed_dict[self.feat_inputs] = testb_feat[batch]
                        pred.extend(self.predicted_label.eval(feed_dict=feed_dict).tolist())
                    for sent in range(len(testb)):
                        for word in range(len(testb[sent])):
                            pred[sent][word] = tags[pred[sent][word]]
                        pred[sent] = pred[sent][:len(testb[sent])]
                    new_conll_sents = data_converter.add_column(testb, data_converter.tags_to_conll(pred))
                    conlleval.evaluate(
                        data_converter.conll_to_data_stream(new_conll_sents, write_to_file="rnn_results.testb"))
                    break

    def get_char_layer(self):
        self.char_embed_placeholder = self.build_char_embedding()
        self.char_inputs = tf.placeholder(tf.int32, shape=[None, None, self.max_char])
        self.char_embed = tf.nn.embedding_lookup(self.char_embed_placeholder, self.char_inputs)
        self.word_length = tf.placeholder(tf.int32, [None, None])
        char_shape = tf.shape(self.char_embed)
        char_embed = tf.reshape(self.char_embed, shape=[-1, self.max_char, 300])

        if self.char_method == "BiLSTM":
            cell_fw = tf.contrib.rnn.LSTMCell(self.char_hidden, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(self.char_hidden, state_is_tuple=True)

            word_length = tf.reshape(self.word_length, shape=[-1])

            _, ((_, output_fw), (_, output_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                                  cell_bw, char_embed,
                                                                                  dtype=tf.float32,
                                                                                  sequence_length=word_length
                                                                                  )
            output_fw_bw = tf.concat([output_fw, output_bw], axis=-1)
            self.char_output = tf.reshape(output_fw_bw, shape=[-1, char_shape[1], 2 * self.char_hidden])
            embed = tf.concat([self.embed, self.char_output], axis=-1)
        else:
            pooled_outputs = list()
            for filter_size in [2, 3, 4, 5]:
                filter_shape = [filter_size, 300, 1, 3]
                b = tf.Variable(tf.constant(0.1, shape=[3]))
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                conv = tf.nn.conv2d(tf.expand_dims(char_embed, axis=3), W, strides=[1, 1, 1, 1], padding="VALID")
                relu = tf.nn.relu(tf.nn.bias_add(conv, b))

                pooled = tf.nn.max_pool(relu, ksize=[1, self.max_char - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID')
                pooled_outputs.append(pooled)

            num_filters_total = 12
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.max_len, num_filters_total])

            embed = tf.concat([self.embed, self.h_pool_flat], axis=-1)
        return embed