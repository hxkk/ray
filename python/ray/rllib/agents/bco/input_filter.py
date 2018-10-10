import numpy

def get_input_size():
    return 169

def get_output_size():
    return 19

def input_filter(obs):
    new_obs = None
    # 412 -> 169
    if isinstance(obs, numpy.ndarray):
        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[0:7 + 1, 9:12 + 1, 14:15 + 1, \
                             17:24 + 1, 26:29 + 1, 31:32 + 1, \
                             51:116 + 1, 150:215 + 1, 403:411 + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[0:7 + 1, 9:12 + 1, 14:15 + 1, \
                             17:24 + 1, 26:29 + 1, 31:32 + 1, \
                             51:116 + 1, 150:215 + 1, 403:411 + 1]]
        else:
            print("Input shape error!", obs.shape)
    elif isinstance(obs, list):
        new_obs = []
        # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
        new_obs += obs[0:7 + 1] + obs[9:12 + 1] + obs[14:15 + 1] \
                   + obs[17:24 + 1] + obs[26:29 + 1] + obs[31:32 + 1] \
                   + obs[51:116 + 1] + obs[150:215 + 1] + obs[403:411 + 1]

    return new_obs