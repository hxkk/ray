from ray.rllib.agents.bco.inverse_dynamics_model import InverseDynamicsModel
from osim.env import ProstheticsEnv

def env_creator(env_config):
    env = ProstheticsEnv(False)
#     env = gym.make("CartPole-v0")
    return env

config={
    # Discount factor
    "gamma": 0.998,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # Number of workers
    # "num_workers": 0, # 72 * 3 + 5
    "num_workers": 7, # 72 * 3 + 5
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
    "demo_file_path": "/data/nips/demos/subject1_3_v01.xlsx",
    "env_config": {},
    "model": {},
    "grad_clip": 40.0,
    "batch_size": 100,
    "lr": 0.0001,
    "ext_data_path": "",
}


env_model = InverseDynamicsModel(env_creator, config, True)
# env_model = InverseDynamicsModel(env_creator, config)
# env_mdoel_data = pickle.load(open("/data/nips/ckpt/checkpoint-386.env_mdoel_data", "rb"))
# env_mdoel_data = pickle.load(open("/data/nips/ckpt/env_mdoel_data.test"+str(STATE_SIZE), "rb"))
# env_model.set_weights(env_mdoel_data)

env_model.train_ext_data()
# env_model.temp_test()