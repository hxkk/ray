from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import ray
from ray.rllib.agents.agent import Agent, with_common_config
from ray.rllib.agents.bco.bco_policy_graph import BCOPolicyGraph
from ray.rllib.optimizers.model_based_sync_samples_optimizer import ModelBased_SyncSamplesOptimizer
from ray.rllib.utils import merge_dicts
from ray.tune.trial import Resources

from ray.rllib.agents.bco.inverse_dynamics_model import InverseDynamicsModel

# from ray.rllib.models.preprocessors import Preprocessor
# from ray.rllib.models import ModelCatalog
#
# class MyPreprocessorClass(Preprocessor):
#     def _init(self):
#         self.shape = self._obs_space.shape
#
#     def transform(self, observation):
#         # pelvis_tilt	pelvis_list	pelvis_rotation	pelvis_tx	pelvis_ty	pelvis_tz	hip_flexion_r
#         # hip_adduction_r	hip_rotation_r	knee_angle_r	ankle_angle_r	hip_flexion_l
#         # hip_adduction_l	hip_rotation_l	knee_angle_l	ankle_angle_l	lumbar_extension
#
#         new_opservation = [] # len(observation["joint_pos"])
#         # print(observation["joint_pos"])
#         for k, v in observation["joint_pos"].iteritems():
#             print(k)
#             print(v)
#
#         return new_opservation
#
# ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)

DEFAULT_CONFIG = with_common_config({
    # Number of workers (excluding master)
    "num_workers": 1,
    # Size of rollout batch
    "batch_size": 100,
    # Max global norm for each gradient calculated by worker
    "grad_clip": 40.0,
    # Learning rate
    "lr": 0.0001,
    # Whether to use a GPU for local optimization.
    "gpu": False,
    # Whether to place workers on GPUs
    "use_gpu_for_workers": False,
    # Model and preprocessor options
    "model": {
        # "custom_preprocessor": "my_prep",
        # "custom_options": {}  # extra options to pass to your classes
    },
    # Arguments to pass to the env creator
    "env_config": {},
    "demo_file_path": "/data/nips/demos/subject1_3_v01.xlsx",
})

class BCOAgent(Agent):
    """Behvioral cloning from observation agent.

    """

    _agent_name = "BCO"
    _default_config = DEFAULT_CONFIG
    _policy_graph = BCOPolicyGraph
    _allow_unknown_configs = True

    @classmethod
    def default_resource_request(cls, config):
        cf = merge_dicts(cls._default_config, config)
        return Resources(cpu=1, gpu=0, extra_cpu=cf["num_workers"])

    def _init(self):
        self.local_evaluator = self.make_local_evaluator(
            self.env_creator, self._policy_graph)
        self.remote_evaluators = self.make_remote_evaluators(
            self.env_creator, self._policy_graph, self.config["num_workers"],
            {})
        self.optimizer = ModelBased_SyncSamplesOptimizer(self.local_evaluator,
                                              self.remote_evaluators,
                                              self.config["optimizer"])
        self.env_model = InverseDynamicsModel(self.env_creator, self.config)
        self.optimizer.set_env_model(self.env_model)

    def _train(self):
        prev_steps = self.optimizer.num_steps_sampled
        self.optimizer.step()
        result = self.optimizer.collect_metrics()
        result.update(timesteps_this_iter=self.optimizer.num_steps_sampled -
                      prev_steps)
        print(result)
        return result

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "checkpoint-{}".format(self.iteration))
        agent_state = ray.get(
            [a.save.remote() for a in self.remote_evaluators])
        extra_data = [self.local_evaluator.save(), agent_state]
        pickle.dump(extra_data, open(checkpoint_path + ".extra_data", "wb"))

        env_mdoel_data = self.env_model.get_weights()
        pickle.dump(env_mdoel_data, open(checkpoint_path + ".env_mdoel_data", "wb"))
        return checkpoint_path

    def _restore(self, checkpoint_path):
        extra_data = pickle.load(open(checkpoint_path + ".extra_data", "rb"))
        self.local_evaluator.restore(extra_data[0])
        ray.get([
            a.restore.remote(o)
            for (a, o) in zip(self.remote_evaluators, extra_data[1])
        ])

        env_mdoel_data = pickle.load(open(checkpoint_path + ".env_mdoel_data", "rb"))
        self.env_model.set_weights(env_mdoel_data)