import ray
from ray.tune.registry import register_env
from ray.rllib.agents import bco

from osim.env import ProstheticsEnv

import pickle

# def env_creator(env_config):
#     return gym.make("CartPole-v0")  # or return your own custom env


def env_creator(env_config):
    # env = ProstheticsEnv(False, integrator_accuracy=3e-2)
    # env = ProstheticsEnv(True)
    env = ProstheticsEnv(False)
    print(env.action_space)
    print(env.action_space.low)
    # env.action_space = gym.spaces.Tuple([gym.spaces.Discrete(11) for _ in range(19)])
    return env

env_creator_name = "custom_env"
register_env(env_creator_name, env_creator) # Register custom env
env = env_creator({})
print("env.action_space ", env.action_space)          # Returns `Box(19,)`
# print("env.action_space.low ", env.action_space.low)      # Returns list of 19 zeroes
# print("env.action_space.high ", env.action_space.high)     # Returns list of 19 ones

# connect to redis server.
ray.init()
# agent = ppo.PPOAgent(env=env_creator_name, config={
#     # "env_config": {},  # config to pass to env creator
#     "simple_optimizer": True
# })

agent = bco.BCOAgent(env=env_creator_name, config={
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
    "invese_model":
    {
        "demo_file_path": "/data/nips/demos/pros_obs169_181008.xlsx",
        "ext_train_file_path": "/data/nips/train_list/train_201810042003.csv",
        "ext_valid_file_path": "/data/nips/train_list/valid_201810042003.csv",
        "ext_output_path" : "/data/nips/ckpt/env_mdoel_data.test",
    },
    # Override model config
    # "model": {
    #     # Whether to use LSTM model
    #     "use_lstm": True,
    #     # Max seq length for LSTM training.
    #     "max_seq_len": 40,
    #     "fcnet_hiddens": [256, 256],
    #     "lstm_cell_size": 256
})

MAX_ITERATIONS = 1000000

# agent.restore('/data/nips/ckpt/checkpoint-1117')
env_mdoel_data = pickle.load(open("/data/nips/ckpt/env_mdoel_data.test169", "rb"))
agent.env_model.set_weights(env_mdoel_data)
print("\m\mhkkkkkkkk  !!!!!!!!!!!!!!!!!!! loaded ", "/data/nips/ckpt/env_mdoel_data.test175")

for i in range(MAX_ITERATIONS):
    result = agent.train() # train agent a iteration
    # print(result)

    if i % 5 == 0:
        agent.save("/data/nips/ckpt/") # save agent network parameters
