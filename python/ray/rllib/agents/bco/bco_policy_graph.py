from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import ray
from ray.rllib.evaluation.tf_policy_graph import PolicyGraph
from ray.rllib.agents.bc.policy import BCPolicy

class BCOPolicyGraph(BCPolicy, PolicyGraph):
    """ BCO poloicy graph

    Actually, this graph is for BC
    """
    def __init__(self, obs_space, action_space, config):
        self.observation_space = obs_space
        super(BCOPolicyGraph, self).__init__(obs_space, action_space, config)

    def initialize(self):
        if self.summarize:
            bs = tf.to_float(tf.shape(self.x)[0])
            tf.summary.scalar("model/policy_loss", self.pi_loss / bs)
            tf.summary.scalar("model/grad_gnorm", tf.global_norm(self.grads))
            tf.summary.scalar("model/var_gnorm", tf.global_norm(self.var_list))
            self.summary_op = tf.summary.merge_all()

        # TODO(rliaw): Can consider exposing these parameters
        self.sess = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2,
                gpu_options=tf.GPUOptions(allow_growth=True)))
        self.variables = ray.experimental.TensorFlowVariables(
            self.loss, self.sess)
        self.sess.run(tf.global_variables_initializer())

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        is_training=False,
                        episodes=None):
        actions = self.sess.run(self.sample, feed_dict={self.x: obs_batch})
        return actions, [], {}

    def compute_gradients(self, samples):
        info = {}
        feed_dict = {
            self.x: samples["obs"],
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
        info["num_samples"] = len(samples.items())
        info["loss"] = loss
        return grad, info