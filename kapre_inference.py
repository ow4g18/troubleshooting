import glob
import os

import tensorflow as tf
import numpy as np
import soundfile as sf

batch_size = 32
num_classes = 1
FEAT_SIZE = (16, 65)

def get_test_dataset(tfrecords,
                batch_size,
                input_size=1024,
                n_classes=1,
                shuffle=False,
                fake_input=False):
    """ Read and preprocess tfrecords into a tf.data.Dataset """

    def parse_func(example_proto):
        """ Parse tfrecords into tf.Feature, to be made into a dataset """

        feature_dict = {
            'signal/id': tf.io.FixedLenFeature([], tf.string),
            'segment/start': tf.io.FixedLenFeature([], tf.int64),
            'segment/end': tf.io.FixedLenFeature([], tf.int64),
            'subsegment/id': tf.io.FixedLenFeature([], tf.int64),
            'subsegment/length': tf.io.FixedLenFeature([], tf.int64),
            'subsegment/signal': tf.io.FixedLenFeature([input_size],
                                                       tf.float32),
            'subsegment/features': tf.io.FixedLenFeature(
                [FEAT_SIZE[0] * FEAT_SIZE[1]], tf.float32),
            'subsegment/label': tf.io.FixedLenFeature([], tf.int64)
        }

        parsed_feature = tf.io.parse_single_example(serialized=example_proto,
                                                    features=feature_dict)

        signal = parsed_feature['subsegment/signal']
        signal = tf.cast(signal, dtype=tf.float32)
        signal = tf.expand_dims(signal, 0)

        labels = parsed_feature['subsegment/label']
        labels = tf.expand_dims(labels, axis=-1)
        labels = tf.cast(labels, dtype=tf.float32)

        return signal, labels

    files = tf.data.Dataset.list_files(tfrecords)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_func, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)

    return dataset

wd = os.getcwd()

model = tf.keras.models.load_model(wd)

tfrecords = [os.path.join(wd, '00000_00256.tfrecord')]

test_audio = get_test_dataset(tfrecords, batch_size=batch_size)

print(model.evaluate(test_audio, batch_size=batch_size))

