import os
import glob
import json
import numpy as np
import argparse

import tensorflow as tf

from time_frequency import Delta
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv1D,
                             Activation,
                             add,
                             Dense,
                             Flatten)
                             
from tensorflow.keras.metrics import (Precision,
                                      Recall,
                                      TruePositives,
                                      TrueNegatives,
                                      FalsePositives,
                                      FalseNegatives)

parser = argparse.ArgumentParser(description='Keras implementation')
parser.add_argument('--data-dir', '-d', type=str,
                    default='/Users/Ollie/Downloads/LibriSpeech/tfrecords/',
                    help='tf records data directory')
parser.add_argument('--model-dir', '-m', type=str,
                    default='/Users/Ollie/Downloads/',
                    help='Directory for model output')
parser.add_argument('--epochs', '-e', type=int,
                    default=10,
                    help='Number of training epochs')

args = parser.parse_args()

batch_size = 32
num_classes = 1
FEAT_SIZE = (16, 65)

# Layer dimensions, taken from defaults of tf implementation
filters = [32, 64, 128]
kernels = [8, 5, 3]
fc = [2048, 2048]

def get_dataset(tfrecords,
                batch_size,
                epochs,
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

    dataset = dataset.repeat(epochs)
    dataset = dataset.map(parse_func, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)

    return dataset

def get_test_dataset(tfrecords,
                batch_size,
                input_size=1024,
                n_classes=1,
                shuffle=False,
                fake_input=False):
    """ Same as get_dataset but without repeating """

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

def dataset_data(dataset):
    """ Used to read the tf.dataset data into memory for troubleshooting purposes """
    data = []
    for i in dataset:
        data.append(tf.convert_to_tensor(i[0]))
    return data

def dataset_labels(dataset):
    """ Used to read the tf.dataset labels into memory for troubleshooting purposes """
    labels = []
    for i in dataset:
        labels.append(tf.convert_to_tensor(i[1]))
    return labels

class LogMelSpectrogram(tf.keras.layers.Layer):
    """ Compute log-magnitude mel-scaled spectrograms """

    def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                 f_min=0.0, f_max=None, **kwargs):
        super(LogMelSpectrogram, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate / 2
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=fft_size // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max)

    def build(self, input_shape):
        self.non_trainable_weights.append(self.mel_filterbank)
        super(LogMelSpectrogram, self).build(input_shape)

    def call(self, waveforms):
        """Forward pass.
        Parameters
        ----------
        waveforms : tf.Tensor, shape = (None, n_samples)
            A Batch of mono waveforms.
        Returns
        -------
        log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
            The corresponding batch of log-mel-spectrograms
        """
        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        def power_to_db(magnitude, amin=1e-16, top_db=80.0):
            """
            https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
            """
            ref_value = tf.reduce_max(magnitude)
            log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
            log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
            log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

            return log_spec

        spectrograms = tf.signal.stft(waveforms,
                                      frame_length=self.fft_size,
                                      frame_step=self.hop_size,
                                      pad_end=True)

        magnitude_spectrograms = tf.abs(spectrograms)

        mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                     self.mel_filterbank)

        log_mel_spectrograms = power_to_db(mel_spectrograms)

        # # add channel dimension
        log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

        return log_mel_spectrograms

    def get_config(self):
        config = {
            'fft_size': self.fft_size,
            'hop_size': self.hop_size,
            'n_mels': self.n_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
        }
        config.update(super(LogMelSpectrogram, self).get_config())

        return config


class ResnetBlock(Model):
    """ Define a single residual block """

    def __init__(self, n_filters, n_kernels, is_training=False):
        super(ResnetBlock, self).__init__()

        self.is_training = is_training
        self.n_filters = n_filters
        self.n_kernels = n_kernels

        self.conv1 = Conv1D(self.n_filters, self.n_kernels[0],
                            activation=None,
                            padding='same',
                            name='conv1')

        self.relu1 = Activation(activation='relu', name='relu1')

        self.conv2 = Conv1D(self.n_filters, self.n_kernels[1],
                            activation=None,
                            padding='same',
                            name='conv2')

        self.relu2 = Activation(activation='relu', name='relu2')

        self.conv3 = Conv1D(self.n_filters, self.n_kernels[2],
                            activation=None,
                            padding='same',
                            name='conv3')

        self.shortcut = Conv1D(self.n_filters, 1,
                               activation=None,
                               padding='same',
                               name='shortcut')

        self.out_block = Activation(activation='relu', name='out_block')

    def call(self, inputs, training=None, mask=None):
        """ Feed forward order """

        x = self.conv1(inputs)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        shortcut = self.shortcut(inputs)

        x = add([x, shortcut])
        out_block = self.out_block(x)

        return out_block


class ResNet(Model):
    """ Construct residual network from residual blocks """

    def __init__(self, n_filters, n_kernels, n_fc):
        super(ResNet, self).__init__()

        self.n_filters = n_filters
        self.n_kernels = n_kernels
        self.n_fc = n_fc

        self.mel = LogMelSpectrogram(sample_rate=16000,
                                     fft_size=512,
                                     hop_size=16,
                                     n_mels=5)
        
        self.delta = Delta(win_length=9, data_format='channels_first')

        # Resnet Blocks
        self.block1 = ResnetBlock(self.n_filters[0],
                                  self.n_kernels)
        self.block2 = ResnetBlock(self.n_filters[1],
                                  self.n_kernels)
        self.block3 = ResnetBlock(self.n_filters[2],
                                  self.n_kernels)
        self.block4 = ResnetBlock(self.n_filters[2],
                                  self.n_kernels)

        # Flatten
        self.flatten = Flatten(name='flatten')

        # FC
        self.fc1 = Dense(self.n_fc[0], activation='relu', name='fc1')
        self.fc2 = Dense(self.n_fc[1], activation='relu', name='fc2')
        self.fc3 = Dense(1, activation='sigmoid', name='fc3')

    def call(self, inputs):

        mel = self.mel(tf.squeeze(inputs, axis=1))


        mel = tf.transpose(mel, perm=[0, 3, 1, 2])


        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(mel)


        # Calculate RMS of spectrogram
        s_sq = tf.math.square(mel)
        sq_sum = tf.math.reduce_sum(s_sq, axis=3, keepdims=True)
        rms = tf.math.sqrt(sq_sum)
        
        delta = self.delta(mfcc)
        delta2 = self.delta(delta)

        cat = tf.concat([mfcc, delta, delta2, rms], axis=3)
        cat = tf.squeeze(cat, axis=1)


        out_block1 = self.block1(cat)

        out_block2 = self.block2(out_block1)

        out_block3 = self.block3(out_block2)

        out_block4 = self.block4(out_block3)


        X = self.flatten(out_block4)
        X = self.fc1(X)
        X = self.fc2(X)

        output = self.fc3(X)

        return output


def main():
    # tf.compat.v1.disable_eager_execution()

    tfrecords_train = glob.glob('{}train/*.tfrecord'.format(args.data_dir))
    tfrecords_val = glob.glob('{}val/*.tfrecord'.format(args.data_dir))
    tfrecords_test = glob.glob('{}test/*.tfrecord'.format(args.data_dir))

    train_dataset = get_dataset(tfrecords_train, batch_size, 1)
    val_dataset = get_dataset(tfrecords_val, batch_size, 1)
    test_audio = get_test_dataset(tfrecords_train, batch_size)

    # Can be used if a single piece of audio/label is needed seperately 
    # X_train = dataset_data(test_audio)
    # X_train = tf.gather_nd(tf.squeeze(X_train, 1), indices=[[100]])   
    # y_train = dataset_labels(test_audio)
    # y_train = tf.gather_nd(tf.squeeze(y_train), indices=[[100]]) 

    model = ResNet(filters, kernels, fc)

    # # Visualise model architecture, save to file
    # tf.keras.utils.plot_model(model,
    #                           to_file=args.model_dir + 'plot.png',
    #                           show_shapes=True,
    #                           expand_nested=True)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001),
                  metrics=['accuracy',
                           Precision(),
                           Recall(),
                           TruePositives(),
                           TrueNegatives(),
                           FalsePositives(),
                           FalseNegatives()],
                  run_eagerly=True)

    history = model.fit(train_dataset,
                        validation_data=train_dataset,
                        epochs=args.epochs,
                        steps_per_epoch=1,
                        validation_steps=1,
                        verbose=2)
    
    print(model.summary())

    # print(model.predict(test_audio))

    # model.save(args.model_dir)

    # # Save Json metrics to file
    # with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
    #     json.dump(str(history.history), f)

if __name__ == "__main__":
    main()
