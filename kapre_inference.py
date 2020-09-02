import glob

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

# # Load the TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="/Users/Ollie/Downloads/kapre_training/keras_output_2/model.tflite")
# interpreter.allocate_tensors()

# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# sound, _ = sf.read('/Users/Ollie/Downloads/021A-C0897X0085XX-AAZZP0.wav')
# for i in range(len(sound // 1024)):
#     audio = sound[i * 1024: (i+1) * 1024]
#     audio = tf.convert_to_tensor(audio, dtype='float32')
#     audio = tf.expand_dims(audio, axis=0)
#     audio = tf.expand_dims(audio, axis=0)
#     interpreter.set_tensor(0, audio)

#     interpreter.invoke()
#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     print(output_data)

# audio = tf.convert_to_tensor(audio[10240:11264], dtype='float32')
# audio = tf.expand_dims(audio, axis=0)
# audio = tf.expand_dims(audio, axis=0)

# audio = np.array([0]*1024, dtype='float32')
# audio = tf.expand_dims(audio, axis=0)
# audio = tf.expand_dims(audio, axis=0)

# print(audio)

# interpreter.set_tensor(0, audio)

# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)


# ------------------------------------------
model_dir = '/Users/Ollie/Downloads/kapre_training/keras_output_2'

model = tf.keras.models.load_model(model_dir)

tfrecords_train = glob.glob('{}train/*.tfrecord'.format('/Users/Ollie/Downloads/LibriSpeech/tfrecords/'))

# sound, _ = sf.read('/Users/Ollie/Downloads/021A-C0897X0085XX-AAZZP0.wav')
# for i in range(len(sound // 1024)):
#     audio = sound[i * 1024: (i+1) * 1024]
#     # audio = np.array([0.] * 1024)
#     audio = tf.expand_dims(audio, axis=0)
#     audio = tf.expand_dims(audio, axis=0)
#     print(audio.shape)

#     print(model.predict(audio, batch_size=1))

test_audio = get_test_dataset(tfrecords_train, batch_size=32)

# for i in test_audio:
#     x = i[0][0]
# x = tf.expand_dims(x, axis=1)

# print(model.predict(x))

print(model.evaluate(test_audio, batch_size=32))

