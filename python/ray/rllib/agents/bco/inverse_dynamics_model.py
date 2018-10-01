from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import gym

import ray
from ray.rllib.models import ModelCatalog

from ray.rllib.models.fcnet import FullyConnectedNetwork

import numpy
import pandas

class InverseDynamicsModel(object):
    def __init__(self, env_creator, config):
        self.local_steps = 0
        self.config = config
        self.summarize = config.get("summarize")
        env = ModelCatalog.get_preprocessor_as_wrapper(env_creator(self.config["env_config"]), self.config["model"])
        self._setup_model(env.observation_space, env.action_space)
        self._setup_loss(env.action_space)
        self.setup_gradients()
        self.initialize()

        self._load_demonstration()

    def _setup_model(self, obs_space, ac_space):
        self.x = tf.placeholder(tf.float32, shape=[None, numpy.prod([2] + list(obs_space.shape))])
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            ac_space, self.config["model"])
        ModelCatalog.register_custom_model("inverse_dynamics_model", FullyConnectedNetwork)
        self._model = ModelCatalog.get_model(self.x, logit_dim,
                                                options={"custom_model": "inverse_dynamics_model"})
        self.logits = self._model.outputs
        self.curr_dist = dist_class(self.logits)
        self.sample = self.curr_dist.sample()
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          tf.get_variable_scope().name)

    def _setup_loss(self, action_space):
        if isinstance(action_space, gym.spaces.Box):
            self.ac = tf.placeholder(
                tf.float32, [None] + list(action_space.shape), name="ac")
        elif isinstance(action_space, gym.spaces.Discrete):
            self.ac = tf.placeholder(tf.int64, [None], name="ac")
        else:
            raise NotImplementedError("action space" +
                                      str(type(action_space)) +
                                      "currently not supported")
        log_prob = self.curr_dist.logp(self.ac)
        self.pi_loss = -tf.reduce_sum(log_prob)
        self.loss = self.pi_loss

    def setup_gradients(self):
        grads = tf.gradients(self.loss, self.var_list)
        self.grads, _ = tf.clip_by_global_norm(grads, self.config["grad_clip"])
        grads_and_vars = list(zip(self.grads, self.var_list))
        opt = tf.train.AdamOptimizer(self.config["lr"])
        self._apply_gradients = opt.apply_gradients(grads_and_vars)

    def initialize(self):
        if self.summarize:
            bs = tf.to_float(tf.shape(self.x)[0])
            tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
            tf.summary.scalar("model/grad_gnorm", tf.global_norm(self.grads))
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

    def compute_gradients(self, samples):
        info = {}

        transition_state = numpy.stack([samples["obs"], samples["new_obs"]], axis=2)
        transition_state = transition_state.reshape(transition_state.shape[0], -1)
        feed_dict = {
            # self.x: samples["observations"],
            self.x: transition_state,
            self.ac: samples["actions"]
        }
        self.grads = [g for g in self.grads if g is not None]
        self.local_steps += 1
        if self.summarize:
            loss, grad, summ = self.sess.run(
                [self.loss, self.grads, self.summary_op], feed_dict=feed_dict)
            info["summary"] = summ
        else:
            loss, grad = self.sess.run(
                [self.loss, self.grads], feed_dict=feed_dict)
        info["num_samples"] = len(samples["obs"])
        info["loss"] = loss
        return grad, info

    def apply_gradients(self, grads):
        feed_dict = {self.grads[i]: grads[i] for i in range(len(grads))}
        self.sess.run(self._apply_gradients, feed_dict=feed_dict)

    def compute_apply(self, samples):
        """Fused compute gradients and apply gradients call.

        Returns:
            grad_info: dictionary of extra metadata from compute_gradients().
            apply_info: dictionary of extra metadata from apply_gradients().

        Examples:
            >>> batch = ev.sample()
            >>> ev.compute_apply(samples)
        """

        grads, grad_info = self.compute_gradients(samples)
        apply_info = self.apply_gradients(grads)
        return grad_info, apply_info

    # def _infer_actions(self, samples):
    #
    #     return actions

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        is_training=False,
                        episodes=None):
        actions = self.sess.run(self.sample, feed_dict={self.x: obs_batch})
        return actions

    def _load_demonstration(self):
        df = pandas.read_excel(self.config["demo_file_path"])
        df = df.loc[:, "pelvis_tilt":]

        self.demo = df.as_matrix()

    def _get_demo_transitions(self):

        demo_obs_array = self.demo[:(self.demo.shape[0]-1)].copy()
        demo_new_obs_array = self.demo[1:].copy()
        # print("demo_obs_array", demo_obs_array.shape)
        # print("demo_obs_array", demo_obs_array)
        # print("demo_new_obs_array", demo_new_obs_array.shape)
        # print("demo_new_obs_array", demo_new_obs_array)
        demo_transitioins = numpy.concatenate((demo_obs_array, demo_new_obs_array), axis=1)
        # print(demo_transitioins.shape)
        # transition_state = numpy.stack([demo_obs_array, demo_new_obs_array], axis=2)
        # transition_state = transition_state.reshape(transition_state.shape[0], -1)
        return demo_transitioins

    def get_weights(self):
        weights = self.variables.get_weights()
        return weights

    def set_weights(self, weights):
        self.variables.set_weights(weights)

    def process(self, samples):
        self.compute_apply(samples)

        demo_transitioins = self._get_demo_transitions()

        actions = self.compute_actions(demo_transitioins)

        new_samples = {"obs":self.demo[: (self.demo.shape[0] - 1)].copy().tolist(), "actions":actions.tolist()}
        return new_samples