import tensorflow as tf
import FWRNNCell
from tensorflow.contrib import legacy_seq2seq
import utils

import numpy as np


class FWGraph(object):
    def __init__(self, config=None):
        self.configuration = config
        self.build_graph()
        self.load_validation()

    def vxn_tests(self):
        reader = utils.DataReader(
                                  data_filename="input_seqs_validation",
                                  batch_size=16)
        input_batch, output = reader.read(False, 1)
        init_op = tf.group(tf.initialize_all_variables(),
                           tf.initialize_local_variables())

        session = tf.Session()
        session.run(init_op)
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session,
                                               coord=coordinator)
        self.vxn_input = []
        self.vxn_answers = []
        try:
            while not coordinator.should_stop():
                input_data, ans = session.run([input_batch, output])
                self.vxn_input.append(input_data)
                self.vxn_answers.append(ans)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coordinator.request_stop()
        coordinator.join(threads)
        session.close()

        tmp = [-1, self.configuration.input_length]
        self.vxn_input = np.array(self.vxn_input).reshape(tmp)
        self.vxn_answers = np.array(self.vxn_answers).reshape([-1, 1])

    def build_graph(self):
        config = self.configuration
        self.reader = utils.DataReader(
                                       seq_len=config.seq_length,
                                       batch_size=config.batch_size,
                                       data_filename=config.data_filename)

        self.cell = FWRNNCell(num_units=config.rnn_size)

        self.input_data = tf.placeholder(tf.int32, [None, config.input_length])
        self.answers = tf.placeholder(tf.int32, [None, 1])
        self.initial_state = self.cell.zero_state(
                                                  tf.shape(self.answers)[0],
                                                  tf.float32)
        self.fw_initial = self.cell.fw_zero(
                                            tf.shape(self.answers)[0],
                                            tf.float32)

        with tf.variable_scope("emb_input"):
            embedding = tf.get_variable("emb",
                                        [config.size_chars,
                                         config.embedding_size])
            inputs = tf.split(
                              tf.nn.embedding_lookup(embedding,
                                                     self.input_data),
                              config.input_length,
                              1)
            inputs = [tf.squeeze(input, [1]) for input in inputs]

        with tf.variable_scope("rnn_desig"):
            state = (self.initial_state, self.fw_initial)
            output = None

            for i, input in enumerate(inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = self.cell(input, state)

        with tf.variable_scope("softmax"):
            softmax_w = tf.get_variable(
                                      "softmax_w",
                                      [config.rnn_size,
                                       config.size_chars])
            softmax_b = tf.get_variable("softmax_b", [config.size_chars])
            self.logits = tf.matmul(output, softmax_w) + softmax_b
            self.p = tf.nn.softmax(self.logits)
            self.output = tf.cast(tf.reshape(
                                             tf.arg_max(self.p, 1),
                                             [-1, 1]),
                                  tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.output, self.answers), tf.float32))

            loss = legacy_seq2seq.sequence_loss_by_example([self.logits],
                                            [tf.reshape(self.answers, [-1])],
                                            [tf.ones([config.batch_size])],
                                            config.size_chars)

        self.cost = tf.reduce_mean(loss)
        self.end_state = state

        train_vars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost,
                                                           train_vars),
                                              config.grad_clip)
        optimizer = tf.train.AdamOptimizer()
        self.train_op = optimizer.apply_gradients(zip(gradients, train_vars))

        self.summary_accuracy = tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('cost', self.cost)
        self.summary_all = tf.summary.merge_all()
