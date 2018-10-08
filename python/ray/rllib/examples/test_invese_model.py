from ray.rllib.agents.bco.inverse_dynamics_model import InverseDynamicsModel

from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)
# env.change_model(model='3D', prosthetic=False)

print(env.action_space)          # Returns `Box(19,)`
print(env.action_space.low)      # Returns list of 19 zeroes
print(env.action_space.high)     # Returns list of 19 ones


env_model = InverseDynamicsModel(env_creator, config, True)
env_mdoel_data = pickle.load(open("/data/nips/ckpt/checkpoint-1128.env_mdoel_data", "rb"))
env_model.set_weights(env_mdoel_data)

actions = env_model.test_model()

observation = env.reset()

for i in range(300):
    # action = df.loc[i][1:].tolist()
    action = actions[i]
    observation, reward, done, info = env.step(action)
    print(reward)