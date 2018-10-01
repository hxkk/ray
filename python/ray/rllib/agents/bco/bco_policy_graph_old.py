from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import ray
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.evaluation.tf_policy_graph import TFPolicyGraph


class BCLoss(object):
    def __init__(self, action_dist, actions):
        self.loss = -tf.reduce_sum(action_dist.logp(actions))


class BCOPolicyGraph(TFPolicyGraph):
    """ BCO poloicy graph

    Actually, this graph is for BC
    """
    def __init__(self, obs_space, action_space, config):
        config = dict(ray.rllib.agents.pg.pg.DEFAULT_CONFIG, **config)
        self.config = config

        # Setup policy
        obs = tf.placeholder(tf.float32, shape=[None] + list(obs_space.shape))
        dist_class, self.logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])
        self.model = ModelCatalog.get_model(
            obs, self.logit_dim, options=self.config["model"])
        action_dist = dist_class(self.model.outputs)  # logit for each action

        # Setup policy loss
        actions = ModelCatalog.get_action_placeholder(action_space)
        advantages = tf.placeholder(tf.float32, [None], name="adv")
        loss = BCLoss(action_dist, actions).loss

        # Initialize TFPolicyGraph
        sess = tf.get_default_session()
        loss_in = [
            ("obs", obs),
            ("actions", actions),
        ]

        TFPolicyGraph.__init__(
            self,
            obs_space,
            action_space,
            sess,
            obs_input=obs,
            action_sampler=action_dist.sample(),
            loss=loss,
            loss_inputs=loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            seq_lens=self.model.seq_lens,
            max_seq_len=config["model"]["max_seq_len"])
        sess.run(tf.global_variables_initializer())

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None):
        return compute_advantages(
            sample_batch, 0.0, self.config["gamma"], use_gae=False)

    def get_initial_state(self):
        return self.model.state_init

    def compute_gradients(self, samples):
        info = {}
        feed_dict = {
            "": samples["observations"],
            self.ac: samples["infer_actions"]
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
        info["num_samples"] = len(samples)
        info["loss"] = loss
        return grad, info