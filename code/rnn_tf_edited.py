import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams["toolbar"] = "None"

num_epochs = 100
total_series_length = 25000
truncated_backprop_length = 15  # recommended by wildml
state_size = 4
num_classes = 2
repeat_delay_count = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length


def create_toy_data():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, repeat_delay_count)
    y[0:repeat_delay_count] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)

query_placeholder = tf.placeholder(
                                    tf.float32,
                                    [batch_size,
                                        truncated_backprop_length])

answer_placeholder = tf.placeholder(
                                    tf.int32,
                                    [batch_size,
                                     truncated_backprop_length])

initial_placeholder = tf.placeholder(tf.float32, [batch_size, state_size])

W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1, state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, num_classes)), dtype=tf.float32)

inputs_series = tf.unstack(query_placeholder, axis=1)
labels_series = tf.unstack(answer_placeholder, axis=1)

current_state = initial_placeholder
states_series = []

for current_input in inputs_series:
    current_input = tf.reshape(current_input, [batch_size, 1])
    input_and_state_concatenated = tf.concat(
                                            axis=1,
                                            values=[
                                                    current_input,
                                                    current_state])

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)
    states_series.append(next_state)
    current_state = next_state

# These are not normalized
out_series = [tf.matmul(state, W2) + b2 for state in states_series]
predictions_series = [tf.nn.softmax(layer) for layer in out_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                logits=layer,
                                                labels=labels)
                                                for layer, labels in zip(
                                                    out_series,
                                                    labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot_error(loss_list, predictions_series, batchX, batchY):
    plt.cla()
    plt.plot(loss_list)

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x, y = create_toy_data()
        _current_state = np.zeros((batch_size, state_size))

        print("Epoch: ", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:, start_idx:end_idx]
            batchY = y[:, start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [
                    total_loss,
                    train_step,
                    current_state,
                    predictions_series],
                feed_dict={
                    query_placeholder: batchX,
                    answer_placeholder: batchY,
                    initial_placeholder: _current_state
                })

            loss_list.append(_total_loss)

            if batch_idx % 100 == 0:
                print("Step", batch_idx, "Error", _total_loss)
                plot_error(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()
