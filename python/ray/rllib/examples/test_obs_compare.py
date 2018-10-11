import numpy as np
from osim.env import ProstheticsEnv
import pandas

def flatten(d):
    res = []  # Result list
    if isinstance(d, dict):
        for key, val in d.items():
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        res = [d]

    return np.array(res)


params = np.load('/data/nips/params_1520.npy')

weight, mean, std = params[0], params[1].rs.mean, params[1].rs.std

env = ProstheticsEnv(False)
# env.osim_model.list_elements()

obs = env.reset(False)
total_reward = 0
done = False
step = 1

time = 0
action_list = []
obs_list = []
print("=============================")
print([time], obs["joint_pos"])
join_list = []
while not done:
   join_list.append(obs["joint_pos"])
   time = step * 0.01
   obs = flatten(obs)
   obs_list.append([time] + obs.tolist())
   # print(time)
   # print([time] + obs.tolist())
   action = np.dot(weight, (obs - mean)/std)
   action = [i / 10 for i in action]  # for discretized actions
   action_list.append([time] + action)
   obs, reward, done, _ = env.step(action, False)
   print("=============================")
   print([time], obs["joint_pos"])
   print(step, 'Reward:', reward)
   step += 1
   total_reward += reward
df = pandas.DataFrame(obs_list)
df.to_csv("/data/nips/es_obs.csv", index=False)

ddf = pandas.DataFrame(join_list)
ddf.to_csv("/data/nips/es_joint_obs.csv", index=False)

# df = pandas.DataFrame(action_list, columns=["time", "abd_r", "add_r", "hamstrings_r",
#                                "bifemsh_r", "glut_max_r", "iliopsoas_r",
#                                "rect_fem_r", "vasti_r", "abd_l", "add_l",
#                                "hamstrings_l", "bifemsh_l", "glut_max_l",
#                                "iliopsoas_l", "rect_fem_l", "vasti_l",
#                                "gastroc_l", "soleus_l", "tib_ant_l"])
# df.to_csv("/data/nips/es_control.csv", index=False)
# env.osim_model.model.printControlStorage("control.sto")
# env.osim_model.manager.getStateStorage().printToFile("states.sto", "w")