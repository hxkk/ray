from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gym
import glob

import ray
from ray.rllib.models import ModelCatalog

from ray.rllib.models.fcnet import FullyConnectedNetwork
from ray.rllib.agents.bco.external_data_provider import input_fn

import numpy
import pandas
import pickle
import os

class InverseDynamicsModel(object):
    def __init__(self, env_creator, config, is_ext_train=False):
        self.local_steps = 0
        self.config = config
        self.summarize = config.get("summarize")
        env = ModelCatalog.get_preprocessor_as_wrapper(env_creator(self.config["env_config"]), self.config["model"])

        if is_ext_train:
            train_dataset = input_fn(self.config["inverse_model"]["ext_train_file_path"])
            valid_dataset = input_fn(self.config["inverse_model"]["ext_valid_file_path"])
            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes)
            next_element = iterator.get_next()
            self.x = next_element[0]
            self.ac = next_element[1]

            self.training_init_op = iterator.make_initializer(train_dataset)
            self.validation_init_op = iterator.make_initializer(valid_dataset)
        else:
            self.x = tf.placeholder(tf.float32, shape=[None, numpy.prod([2] + list(env.observation_space.shape))])
            if isinstance(env.action_space, gym.spaces.Box):
                self.ac = tf.placeholder(
                    tf.float32, [None] + list(env.action_space.shape), name="ac")
            elif isinstance(env.action_space, gym.spaces.Discrete):
                self.ac = tf.placeholder(tf.int64, [None], name="ac")
            else:
                raise NotImplementedError("action space" +
                                          str(type(env.action_space)) +
                                          "currently not supported")

        # Setup graph
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            env.action_space, self.config["model"])
        self._model = FullyConnectedNetwork(self.x, logit_dim, {})
        self.logits = self._model.outputs
        self.curr_dist = dist_class(self.logits)
        self.sample = self.curr_dist.sample()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

        # Setup loss
        log_prob = self.curr_dist.logp(self.ac)
        self.pi_loss = -tf.reduce_sum(log_prob)
        self.loss = self.pi_loss
        self.optimizer = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.loss)

        # Setup similarity -> cosine similarity
        normalize_sample = tf.nn.l2_normalize(self.sample, 1)
        normalize_ac = tf.nn.l2_normalize(self.ac, 1)
        self.similarity = 1 - tf.losses.cosine_distance(normalize_sample, normalize_ac, dim=1)

        # Initialize
        self.initialize()

    def initialize(self):
        if self.summarize:
            bs = tf.to_float(tf.shape(self.x)[0])
            tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
            tf.summary.scalar("model/var_gnorm", tf.global_norm(self.var_list))
            self.summary_op = tf.summary.merge_all()

        # TODO(rliaw): Can consider exposing these parameters
        self.sess = tf.Session(
            # graph=self.g,
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2,
                gpu_options=tf.GPUOptions(allow_growth=True)))
        self.variables = ray.experimental.TensorFlowVariables(
            self.loss, self.sess)
        self.sess.run(tf.global_variables_initializer())

    def update_model(self, samples):  # should fix
        info = {}

        transition_state = numpy.stack([samples["obs"], samples["new_obs"]], axis=2)
        # print("hkkkk======================================")
        # print(samples["obs"][1])
        # print(samples["new_obs"][0])
        # print(samples["obs"][1] == samples["new_obs"][0])
        # print("===========================================")
        transition_state = transition_state.reshape(transition_state.shape[0], -1)
        feed_dict = {
            # self.x: samples["observations"],
            self.x: transition_state,
            self.ac: samples["actions"]
        }
        self.local_steps += 1
        if self.summarize:
            loss, _, summ = self.sess.run(
                [self.loss, self.optimizer, self.summary_op], feed_dict=feed_dict)
            info["summary"] = summ
        else:
            loss, _ = self.sess.run(
                [self.loss, self.optimizer], feed_dict=feed_dict)
        info["num_samples"] = len(samples["obs"])
        info["loss"] = loss
        return info

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        is_training=False,
                        episodes=None):
        actions = self.sess.run(self.sample, feed_dict={self.x: obs_batch})
        return actions

    def _get_demo_transitions(self):
        demo_list = glob.glob(os.path.join(self.config["inverse_model"]["demo_dir_path"], "*.xlsx"))
        if len(demo_list) == 0:
            print("[env model] get demo error! there is no demo file -> %s",
                  (self.config["inverse_model"]["demo_dir_path"]))
        demo_file = numpy.random.choice(demo_list)

        print("[env model] pick demo: %s" % (demo_file))

        df = pandas.read_excel(demo_file)
        df = df.drop(df.columns[0], axis=1)
        demo = df.as_matrix()

        demo_obs_array = demo[:(demo.shape[0] - 1), :]
        demo_new_obs_array = demo[1:, :]
        # print("demo_obs_array", demo_obs_array.shape)
        # print("demo_obs_array", demo_obs_array)
        # print("demo_new_obs_array", demo_new_obs_array.shape)
        # print("demo_new_obs_array", demo_new_obs_array)
        demo_transitioins = numpy.concatenate((demo_obs_array, demo_new_obs_array), axis=1)
        # print(demo_transitioins.shape)
        # transition_state = numpy.stack([demo_obs_array, demo_new_obs_array], axis=2)
        # transition_state = transition_state.reshape(transition_state.shape[0], -1)
        return demo, demo_transitioins

    def get_weights(self):
        weights = self.variables.get_weights()
        return weights

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def process(self, samples):
        self.update_model(samples)

        demo, demo_transitioins = self._get_demo_transitions()

        infer_actions = self.compute_actions(demo_transitioins)

        new_samples = {"obs": demo[: (demo.shape[0] - 1)].tolist(), "actions": infer_actions.tolist()}
        return new_samples

    def save(self, path):
        env_model_data = self.get_weights()
        pickle.dump(env_model_data, open(path, "wb"))

    def restore(self, path):
        if "ext_input_path" in self.config["inverse_model"] and \
                os.path.isfile(self.config["inverse_model"]["ext_input_path"]):
            env_model_path = self.config["inverse_model"]["ext_input_path"]
        else:
            env_model_path = path

        env_model_data = pickle.load(open(env_model_path, "rb"))
        self.set_weights(env_model_data)

        print("[env model] restore: %s " % (env_model_path))

    def train_ext_data(self, epochs = 100, train_steps = 100000, valid_iters = 20):

        self.sess.run(self.training_init_op)
        self.sess.run(self.validation_init_op)

        for epoch in range(epochs):
            for i in range(train_steps):
                l, _, acc = self.sess.run([self.loss, self.optimizer, self.similarity])
                if i % (train_steps / 10) == 0:
                    print("Epoch: {}, train step: {}, loss: {:.3f}, training similarity: {:.2f}%".format(epoch, i, l,
                                                                                                       acc * 100))
            #                     print("train step: {}, loss: {:.3f}".format(i, l))

            avg_acc = 0
            valid_loss_str = ""
            for i in range(valid_iters):
                acc = self.sess.run([self.similarity])
                avg_acc += acc[0]
            print("Average validation set similarity over {} iterations is {:.2f}%".format(valid_iters,
                                                                                         (avg_acc / valid_iters) * 100))
            self.save(self.config["inverse_model"]["ext_output_path"])

    def test_model(self):
        demo_transitioins = self._get_demo_transitions()

        infer_actions = self.compute_actions(demo_transitioins)

        return infer_actions

    def temp_test(self):
        epochs = 100
        train_steps = 100000
        valid_iters = 20

        self.sess.run(self.training_init_op)
        self.sess.run(self.validation_init_op)

        for epoch in range(epochs):
            for i in range(train_steps):
                l, acc = self.sess.run([self.loss, self.similarity])
                if i % (train_steps / 10) == 0:
                    print("Epoch: {}, train step: {}, loss: {:.3f}, training similarity: {:.2f}%".format(epoch, i, l,
                                                                                                       acc * 100))