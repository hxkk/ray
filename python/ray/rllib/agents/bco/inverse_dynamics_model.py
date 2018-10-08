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
import pickle

# STATE_SIZE = 17
# STATE_SIZE = 51
STATE_SIZE = 175


# STATE_SIZE = 249

def env_creator(env_config):
    env = ProstheticsEnv(False)
    #     env = gym.make("CartPole-v0")
    return env


config = {
    # Discount factor
    "gamma": 0.998,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # Number of workers
    # "num_workers": 0, # 72 * 3 + 5
    "num_workers": 7,  # 72 * 3 + 5
    # GAE(lambda) parameter
    "lambda": 0.95,
    # Initial coefficient for KL divergence
    "num_sgd_iter": 10,
    # Stepsize of SGD
    "sgd_stepsize": 3e-4,
    # batch_size
    "sample_batch_size": 128,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.001,
    # PPO clip parameter
    "clip_param": 0.2,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Number of GPUs to use for SGD
    "num_gpus": 1,
    # Whether to allocate GPUs for workers (if > 0).
    "num_gpus_per_worker": 0,
    # Whether to allocate CPUs for workers (if > 0).
    "num_cpus_per_worker": 1,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "complete_episodes",
    # Which observation filter to apply to the observation
    # "observation_filter": "MeanStdFilter",
    # "observation_filter": "NoFilter",
    # Use the sync samples optimizer instead of the multi-gpu one
    "simple_optimizer": True,
    "demo_file_path": "/data/nips/demos/pros_obs169_181008.xlsx",
    "env_config": {},
    "model": {},
    "grad_clip": 40.0,
    "batch_size": 100,
    "lr": 0.0001,
    "ext_data_path": "",
}

class InverseDynamicsModel(object):
    def __init__(self, env_creator, config, is_ext_train=False):
        self.local_steps = 0
        self.config = config
        self.summarize = config.get("summarize")
        env = ModelCatalog.get_preprocessor_as_wrapper(env_creator(self.config["env_config"]), self.config["model"])

        self._load_demonstration()

        if is_ext_train:
            train_dataset, valid_dataset = self.load_ext_data(self.config["ext_data_path"])
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

        # Setup accuracy -> cosine similarity
        normalize_sample = tf.nn.l2_normalize(self.sample, 1)
        normalize_ac = tf.nn.l2_normalize(self.ac, 1)
        self.accuracy = 1 - tf.losses.cosine_distance(normalize_sample, normalize_ac, dim=1)

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

    def _load_demonstration(self):
        df = pandas.read_excel(self.config["demo_file_path"])
        df = df.drop(df.columns[0], axis=1)

        self.demo = df.as_matrix()

    def _get_demo_transitions(self):
        demo_obs_array = self.demo[:(self.demo.shape[0] - 1), :]
        demo_new_obs_array = self.demo[1:, :]
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
        self.update_model(samples)

        demo_transitioins = self._get_demo_transitions()

        infer_actions = self.compute_actions(demo_transitioins)

        new_samples = {"obs": self.demo[: (self.demo.shape[0] - 1)].tolist(), "actions": infer_actions.tolist()}
        return new_samples

    def load_ext_data(self, path=""):
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

        BASE_PATH = "/data/nips/es_trans_data/"

        #         train_df = pandas.read_csv("/data/nips/train_list/train_201810061612.csv")
        train_df = pandas.read_csv("/data/nips/train_list/train_201810042003.csv")
        train_obs_data = []
        train_act_data = []
        for data in train_df["name"]:
            obs = numpy.load(BASE_PATH + data + "_obs.npy").astype(numpy.float32)
            #             obs_t = obs[:(obs.shape[0]-1), :STATE_SIZE]
            #             obs_next_t = obs[1:, :STATE_SIZE]
            obs = self.input_filter(obs)
            obs_t = obs[:(obs.shape[0] - 1), :]
            obs_next_t = obs[1:, :]
            obs_transitioins = numpy.concatenate((obs_t, obs_next_t), axis=1)
            train_obs_data.append(obs_transitioins)

            act = numpy.load(BASE_PATH + data + "_act.npy").astype(numpy.float32)
            train_act_data.append(act)
        train_obs_data = tf.data.Dataset.from_tensor_slices(numpy.vstack(train_obs_data))
        train_act_data = tf.data.Dataset.from_tensor_slices(numpy.vstack(train_act_data))
        train_dataset = tf.data.Dataset.zip((train_obs_data, train_act_data)).repeat().shuffle(500)
        train_dataset = train_dataset.batch(500)
        train_dataset = train_dataset.map(map_batch)

        #         valid_df = pandas.read_csv("/data/nips/train_list/valid_201810061612.csv")
        valid_df = pandas.read_csv("/data/nips/train_list/valid_201810042003.csv")
        valid_obs_data = []
        valid_act_data = []
        for data in train_df["name"]:
            obs = numpy.load(BASE_PATH + data + "_obs.npy").astype(numpy.float32)
            #             obs_t = obs[:(obs.shape[0]-1), :STATE_SIZE]
            #             obs_next_t = obs[1:, :STATE_SIZE]
            obs = self.input_filter(obs)
            obs_t = obs[:(obs.shape[0] - 1), :]
            obs_next_t = obs[1:, :]
            obs_transitioins = numpy.concatenate((obs_t, obs_next_t), axis=1)
            valid_obs_data.append(obs_transitioins)

            act = numpy.load(BASE_PATH + data + "_act.npy").astype(numpy.float32)
            valid_act_data.append(act)
        valid_obs_data = tf.data.Dataset.from_tensor_slices(numpy.vstack(valid_obs_data))
        valid_act_data = tf.data.Dataset.from_tensor_slices(numpy.vstack(valid_act_data))
        valid_dataset = tf.data.Dataset.zip((valid_obs_data, valid_act_data)).repeat().shuffle(500)
        valid_dataset = valid_dataset.batch(500)
        valid_dataset = valid_dataset.map(map_batch)

        return train_dataset, valid_dataset

    def train_ext_data(self):
        epochs = 100
        train_steps = 100000
        valid_iters = 20

        self.sess.run(self.training_init_op)
        self.sess.run(self.validation_init_op)

        for epoch in range(epochs):
            for i in range(train_steps):
                l, _, acc = self.sess.run([self.loss, self.optimizer, self.accuracy])
                if i % (train_steps / 10) == 0:
                    print("Epoch: {}, train step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(epoch, i, l,
                                                                                                       acc * 100))
            #                     print("train step: {}, loss: {:.3f}".format(i, l))

            avg_acc = 0
            valid_loss_str = ""
            for i in range(valid_iters):
                acc = self.sess.run([self.accuracy])
                avg_acc += acc[0]
            print("Average validation set accuracy over {} iterations is {:.2f}%".format(valid_iters,
                                                                                         (avg_acc / valid_iters) * 100))

            env_mdoel_data = self.get_weights()
            #             pickle.dump(env_mdoel_data, open("/data/nips/ckpt/env_mdoel_data_"+str(epochs), "wb"))
            pickle.dump(env_mdoel_data, open("/data/nips/ckpt/env_mdoel_data.test" + str(STATE_SIZE), "wb"))

    def test_model(self):
        demo_transitioins = self._get_demo_transitions()

        infer_actions = self.compute_actions(demo_transitioins)

        return infer_actions

    def input_filter(self, obs):
        # 412 -> 175
        #         new_obs += obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
#         new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        # 412 -> 169
#         new_obs += obs[0:7 + 1] + obs[9:12 + 1] + obs[14:15 + 1] \
#             + obs[17:24 + 1] + obs[26:29 + 1] + obs[31:32 + 1] \
#             + obs[51:116 + 1] + obs[150:215 + 1] + obs[403:411 + 1]
        new_obs = obs[:, numpy.r_[0:7 + 1, 9:12 + 1, 14:15 + 1,
                                  17:24 + 1, 26:29 + 1, 31:32 + 1,
                                  51:116 + 1, 150:215 + 1, 403:411 + 1]]

        return new_obs

    def temp_test(self):
        epochs = 100
        train_steps = 100000
        valid_iters = 20

        self.sess.run(self.training_init_op)
        self.sess.run(self.validation_init_op)

        for epoch in range(epochs):
            for i in range(train_steps):
                l, acc = self.sess.run([self.loss, self.accuracy])
                if i % (train_steps / 10) == 0:
                    print("Epoch: {}, train step: {}, loss: {:.3f}, training accuracy: {:.2f}%".format(epoch, i, l,
                                                                                                       acc * 100))