import numpy

class joint_pos:
    start = -1
    end = -1

    split0_start = -1
    split0_end = -1
    split1_start = -1
    split1_end = -1
    split2_start = -1
    split2_end = -1


class joint_vel:
    start = -1
    end = -1

    split0_start = -1
    split0_end = -1
    split1_start = -1
    split1_end = -1
    split2_start = -1
    split2_end = -1

class body_pos:
    start = -1
    end = -1


class body_vel:
    start = -1
    end = -1


class body_pos:
    start = -1
    end = -1


class body_vel:
    start = -1
    end = -1


class body_pos_rot:
    start = -1
    end = -1


class body_vel_rot:
    start = -1
    end = -1


class mass_center_pos:
    start = -1
    end = -1


class mass_center_vel:
    start = -1
    end = -1


class mass_center_acc:
    start = -1
    end = -1

def get_input_size():
    return 169

def get_output_size():
    return 19

def input_filter(obs):
    new_obs = None
    # 412 -> 169
    if isinstance(obs, numpy.ndarray):
        joint_pos.start = 0
        joint_pos.end = 15
        joint_pos.split0_start = 0
        joint_pos.split0_end = 7
        joint_pos.split1_start = 9
        joint_pos.split1_end = 12
        joint_pos.split2_start = 14
        joint_pos.split2_end = 15

        joint_vel.start = 17
        joint_vel.end = 32
        joint_vel.split0_start = 17
        joint_vel.split0_end = 24
        joint_vel.split1_start = 26
        joint_vel.split1_end = 29
        joint_vel.split2_start = 31
        joint_vel.split2_end = 32

        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[joint_pos.split0_start:joint_pos.split0_end + 1, \
                             joint_pos.split1_start:joint_pos.split1_end + 1, \
                             joint_pos.split2_start:joint_pos.split2_end + 1, \
                             joint_vel.split0_start:joint_vel.split0_end + 1, \
                             joint_vel.split1_start:joint_vel.split1_end + 1, \
                             joint_vel.split2_start:joint_vel.split2_end + 1, \
                             51:116 + 1, \
                             150:215 + 1, \
                             403:411 + 1]]
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

def target_filter(obs, target="pos"):
    new_obs = None
    # 412 -> 169
    if target == "pos":
        joint_pos.start = 0
        joint_pos.end = 15
        joint_pos.split0_start = 0
        joint_pos.split0_end = 7
        joint_pos.split1_start = 9
        joint_pos.split1_end = 12
        joint_pos.split2_start = 14
        joint_pos.split2_end = 15

        body_pos.start = 51
        body_pos.end = 83

        body_pos_rot.start = 150
        body_pos_rot.end = 182

        mass_center_pos.start = 403
        mass_center_pos.end = 405

        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[joint_pos.split0_start:joint_pos.split0_end + 1, \
                                 joint_pos.split1_start:joint_pos.split1_end + 1, \
                                 joint_pos.split2_start:joint_pos.split2_end + 1, \
                                 body_pos.start:body_pos.end + 1, \
                                 body_pos_rot.start:body_pos_rot.end + 1, \
                                 mass_center_pos.start: mass_center_pos.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[joint_pos.split0_start:joint_pos.split0_end + 1, \
                                 joint_pos.split1_start:joint_pos.split1_end + 1, \
                                 joint_pos.split2_start:joint_pos.split2_end + 1, \
                                 body_pos.start:body_pos.end + 1, \
                                 body_pos_rot.start:body_pos_rot.end + 1, \
                                 mass_center_pos.start: mass_center_pos.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs += obs[joint_pos.split0_start:joint_pos.split0_end + 1] + \
                       obs[joint_pos.split1_start:joint_pos.split1_end + 1] + \
                       obs[joint_pos.split2_start:joint_pos.split2_end + 1] + \
                       obs[body_pos.start:body_pos.end + 1] + \
                       obs[body_pos_rot.start:body_pos_rot.end + 1] + \
                       obs[mass_center_pos.start: mass_center_pos.end + 1]
    elif target == "vel":
        joint_vel.start = 17
        joint_vel.end = 32
        joint_vel.split0_start = 17
        joint_vel.split0_end = 24
        joint_vel.split1_start = 26
        joint_vel.split1_end = 29
        joint_vel.split2_start = 31
        joint_vel.split2_end = 32

        body_vel.start = 84
        body_vel.end = 116

        body_vel_rot.start = 183
        body_vel_rot.end = 215

        mass_center_vel.start = 406
        mass_center_vel.end = 408

        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[joint_vel.split0_start:joint_vel.split0_end + 1, \
                                 joint_vel.split1_start:joint_vel.split1_end + 1, \
                                 joint_vel.split2_start:joint_vel.split2_end + 1, \
                                 body_vel.start:body_vel.end + 1, \
                                 body_vel_rot.start:body_vel_rot.end + 1, \
                                 mass_center_vel.start: mass_center_vel.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[joint_vel.split0_start:joint_vel.split0_end + 1, \
                              joint_vel.split1_start:joint_vel.split1_end + 1, \
                              joint_vel.split2_start:joint_vel.split2_end + 1, \
                              body_vel.start:body_vel.end + 1, \
                              body_vel_rot.start:body_vel_rot.end + 1, \
                              mass_center_vel.start: mass_center_vel.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs += obs[joint_vel.split0_start:joint_vel.split0_end + 1] + \
                       obs[joint_vel.split1_start:joint_vel.split1_end + 1] + \
                       obs[joint_vel.split2_start:joint_vel.split2_end + 1] + \
                       obs[body_vel.start:body_vel.end + 1] + \
                       obs[body_vel_rot.start:body_vel_rot.end + 1] + \
                       obs[mass_center_vel.start: mass_center_vel.end + 1]
    elif target == "acc":
        mass_center_acc.start = 409
        mass_center_acc.end = 411
        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs += obs[mass_center_acc.start:mass_center_acc.end + 1]

    return new_obs

def demo_filter(obs, target="pos"):
    if target == "pos":
        joint_pos.start = 0
        joint_pos.end = 13

        body_pos.start = 28
        body_pos.end = 60

        body_pos_rot.start = 94
        body_pos_rot.end = 126

        mass_center_pos.start = 160
        mass_center_pos.end = 162
        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[joint_pos.start:joint_pos.end + 1,
                                 body_pos.start:body_pos.end + 1,
                                 body_pos_rot.start:body_pos_rot.end + 1, \
                                 mass_center_pos.start:mass_center_pos.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[joint_pos.start:joint_pos.end + 1,
                                 body_pos.start:body_pos.end + 1,
                                 body_pos_rot.start:body_pos_rot.end + 1, \
                                 mass_center_pos.start:mass_center_pos.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs +=  obs[joint_pos.start:joint_pos.end + 1] + \
                        obs[body_pos.start:body_pos.end + 1] + \
                        obs[body_pos_rot.start:body_pos_rot.end + 1] + \
                        obs[mass_center_pos.start:mass_center_pos.end + 1]
    elif target == "vel":
        joint_vel.start = 14
        joint_vel.end = 27

        body_vel.start = 61
        body_vel.end = 93

        body_vel_rot.start = 127
        body_vel_rot.end = 159

        mass_center_vel.start = 163
        mass_center_vel.end = 165
        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[joint_vel.start:joint_vel.end + 1,
                                 body_vel.start:body_vel.end + 1,
                                 body_vel_rot.start:body_vel_rot.end + 1, \
                                 mass_center_vel.start:mass_center_vel.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[joint_vel.start:joint_vel.end + 1,
                             body_vel.start:body_vel.end + 1,
                             body_vel_rot.start:body_vel_rot.end + 1, \
                             mass_center_vel.start:mass_center_vel.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs +=  obs[joint_vel.start:joint_vel.end + 1] + \
                        obs[body_vel.start:body_vel.end + 1] + \
                        obs[body_vel_rot.start:body_vel_rot.end + 1] + \
                        obs[mass_center_vel.start:mass_center_vel.end + 1]
    elif target == "acc":
        mass_center_acc.start = 166
        mass_center_acc.end = 168
        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs += obs[mass_center_acc.start:mass_center_acc.end + 1]

    return new_obs
