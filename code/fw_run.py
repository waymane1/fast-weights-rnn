import tensorflow as tf
import configuration
import numpy as np

from fw_graph import FWGraph


def begin_training(config):
    with tf.Graph().as_default():
        model = FWGraph(config)
        input_batch, output = model.reader.read(shuffle=False, num_epochs=1)

        initialize_all = tf.group(tf.initialize_all_variables(),
                                  tf.initialize_local_variables())
        session = tf.Session()
        session.run(initialize_all)
        save_db = tf.train.Saver(tf.all_variables())
        coordinator = tf.train.Coordinator()

        queues = tf.train.start_queue_runners(sess=session, coord=coordinator)

        save_db.restore(session, "./save/fastweights/save_db_60")

        num_correct = 0
        num_total = 0

        try:
            while not coordinator.should_stop():
                input_data, tg = session.run([input_batch, output])
                probs = session.run(
                                    [model.probs],
                                    {model.input_data: input_data,
                                     model.targets: tg})
                probs = np.array(probs).reshape([-1, config.vocab_size])
                tg = np.array([x[0] for x in tg])
                output = np.argmax(probs, axis=1)

                num_correct += np.sum(output == tg)
                num_total += len(output)
        except tf.errors.OutOfRangeError:
            pass
        finally:
            coordinator.request_stop()
        print("Accuracy: %f" % (float(num_correct) / num_total))
        coordinator.join(queues)
        session.close()


if __name__ == "__main__":
    config = configuration.ModelConfig(data_filename="input_seqs_eval")
    begin_training(config)
