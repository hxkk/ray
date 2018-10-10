import tensorflow as tf
import pandas
import numpy
from ray.rllib.agents.bco.input_filter import *

BASE_PATH = "/data/nips/es_trans_data/"


class OpensimTransitionDataset(tf.data.Dataset):
    def __init__(self, data_path):
        files = pandas.read_csv(data_path)
        files = files["name"].tolist()

        self._dataset = tf.data.Dataset.from_tensor_slices(files)
        # dataset = dataset.map(
        #         lambda filename: tuple(tf.py_func(read_npy_file, [filename], [tf.float32, tf.float32])))
        self._dataset = self._dataset.map(self._parse_function)
        self._dataset = self._dataset.flat_map(self._flat_function)
        self._dataset = self._dataset.map(self._set_shapes)

    def _set_shapes(self, x, y):
        x.set_shape([get_input_size() * 2])
        y.set_shape([get_output_size()])
        return x, y

    def _read_npy_file(self, filename):
        obs = numpy.load(BASE_PATH + filename.decode() + "_obs.npy")
        act = numpy.load(BASE_PATH + filename.decode() + "_act.npy")

        obs = input_filter(obs)
        obs_t = obs[:(obs.shape[0] - 1), :]
        obs_next_t = obs[1:, :]
        obs_transitioins = numpy.concatenate((obs_t, obs_next_t), axis=1)

        return obs_transitioins.astype(numpy.float32), act.astype(numpy.float32)

    def _parse_function(self, filename):
        result_tensors = tf.py_func(func=self._read_npy_file, inp=[filename], Tout=[tf.float32, tf.float32])

        return {"obs": result_tensors[0], "act": result_tensors[1]}

    def _flat_function(self, named_elements):
        obs_data = tf.data.Dataset.from_tensor_slices(named_elements["obs"])
        act_data = tf.data.Dataset.from_tensor_slices(named_elements["act"])

        dataset = tf.data.Dataset.zip((obs_data, act_data))
        return dataset

    # The following four methods are needed to implement a tf.data.Dataset
    # Delegate them to the dataset we create internally
    def _as_variant_tensor(self):
        return self._dataset._as_variant_tensor()

    @property
    def output_classes(self):
        return self._dataset.output_classes

    @property
    def output_shapes(self):
        return self._dataset.output_shapes

    @property
    def output_types(self):
        return self._dataset.output_types


def in_meory_dataset(data_path):
    df = pandas.read_csv(data_path)
    obs_data = []
    act_data = []
    for data in df["name"]:
        obs = numpy.load(BASE_PATH + data + "_obs.npy").astype(numpy.float32)
        obs = input_filter(obs)
        obs_t = obs[:(obs.shape[0] - 1), :]
        obs_next_t = obs[1:, :]
        obs_transitioins = numpy.concatenate((obs_t, obs_next_t), axis=1)
        obs_data.append(obs_transitioins)

        act = numpy.load(BASE_PATH + data + "_act.npy").astype(numpy.float32)
        act_data.append(act)
    obs_data = tf.data.Dataset.from_tensor_slices(numpy.vstack(obs_data))
    act_data = tf.data.Dataset.from_tensor_slices(numpy.vstack(act_data))
    dataset = tf.data.Dataset.zip((obs_data, act_data))

    return dataset


def map_batch(x, y):
    noise_propotion = 0.1
    print('Map batch:')
    print('x shape: %s' % str(x.shape))
    print('y shape: %s' % str(y.shape))
    xnoise = x * noise_propotion * tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.2, dtype=tf.float32)
    #             ynoise = y * noise_propotion * tf.random_normal(shape=tf.shape(y), mean=0.0, stddev=0.2, dtype=tf.float32)
    # Note: this flips ALL images left to right. Not sure this is what you want
    # UPDATE: looks like tf documentation is wrong and you need a 3D tensor?
    # return tf.image.flip_left_right(x), y
    #             return x + xnoise, y + ynoise
    return x + xnoise, y


def input_fn(data_path, in_memory_data=False):
    if in_memory_data:
        dataset = in_meory_dataset(data_path)
    else:
        dataset = OpensimTransitionDataset(data_path)
    dataset = dataset.shuffle(20)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=500)
    dataset = dataset.map(map_batch)

    return dataset