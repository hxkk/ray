import numpy

class slice_point:
    def __init__(self, start=-1, end=-1):
        self.start = start
        self.end = end

def get_input_size():
    return 169

def get_output_size():
    return 19

def input_filter(obs):
    new_obs = None
    # 412 -> 169
    if isinstance(obs, numpy.ndarray):
        # joint_pos = slice_point(start=0, end=15)
        joint_pos0 = slice_point(start=0, end=7)
        joint_pos1 = slice_point(start=9, end=12)
        joint_pos2 = slice_point(start=14, end=15)

        # joint_vel = slice_point(start=17, end=32)
        joint_vel0 = slice_point(start=17, end=24)
        joint_vel1 = slice_point(start=26, end=29)
        joint_vel2 = slice_point(start=31, end=32)

        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[joint_pos0.start:joint_pos0.end + 1, \
                             joint_pos1.start:joint_pos1.end + 1, \
                             joint_pos2.start:joint_pos2.end + 1, \
                             joint_vel0.start:joint_vel0.end + 1, \
                             joint_vel1.start:joint_vel1.end + 1, \
                             joint_vel2.start:joint_vel2.end + 1, \
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
        #joint_pos = slice_point(start=0, end=15)
        joint_pos0 = slice_point(start=0, end=7)
        joint_pos1 = slice_point(start=9, end=12)
        joint_pos2 = slice_point(start=14, end=15)

        body_pos = slice_point(start = 51, end = 83)

        body_pos_rot = slice_point(start=150, end=182)

        mass_center_pos = slice_point(start=403, end=405)

        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[joint_pos0.start:joint_pos0.end + 1, \
                                     joint_pos1.start:joint_pos1.end + 1, \
                                     joint_pos2.start:joint_pos2.end + 1, \
                                     body_pos.start:body_pos.end + 1, \
                                     body_pos_rot.start:body_pos_rot.end + 1, \
                                     mass_center_pos.start: mass_center_pos.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[joint_pos0.start:joint_pos0.end + 1, \
                                     joint_pos1.start:joint_pos1.end + 1, \
                                     joint_pos2.start:joint_pos2.end + 1, \
                                     body_pos.start:body_pos.end + 1, \
                                     body_pos_rot.start:body_pos_rot.end + 1, \
                                     mass_center_pos.start: mass_center_pos.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            # new_obs = obs[0:33+1] + obs[51:116+1] + obs[150:215+1] + obs[403:411+1]
            new_obs += obs[joint_pos0.start:joint_pos0._end + 1] + \
                       obs[joint_pos1.start:joint_pos1._end + 1] + \
                       obs[joint_pos2.start:joint_pos2.end + 1] + \
                       obs[body_pos.start:body_pos.end + 1] + \
                       obs[body_pos_rot.start:body_pos_rot.end + 1] + \
                       obs[mass_center_pos.start: mass_center_pos.end + 1]
    elif target == "vel":
        # joint_vel = slice_point(start=17, end=32)
        joint_vel0 = slice_point(start=17, end=24)
        joint_vel1 = slice_point(start=26, end=29)
        joint_vel2 = slice_point(start=31, end=32)

        body_vel = slice_point(start = 84, end = 116)

        body_vel_rot = slice_point(start=183, end=215)

        mass_center_vel = slice_point(start=406, end=408)

        if isinstance(obs, numpy.ndarray):
            # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[joint_vel0.start:joint_vel0.end + 1, \
                                     joint_vel1.start:joint_vel1.end + 1, \
                                     joint_vel2.start:joint_vel2.end + 1, \
                                     body_vel.start:body_vel.end + 1, \
                                     body_vel_rot.start:body_vel_rot.end + 1, \
                                     mass_center_vel.start: mass_center_vel.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[joint_vel0.start:joint_vel0.end + 1, \
                                     joint_vel1.start:joint_vel1.end + 1, \
                                     joint_vel2.start:joint_vel2.end + 1, \
                                     body_vel.start:body_vel.end + 1, \
                                     body_vel_rot.start:body_vel_rot.end + 1, \
                                     mass_center_vel.start: mass_center_vel.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            new_obs += obs[joint_vel0.start:joint_vel0.end + 1] + \
                       obs[joint_vel1.start:joint_vel1.end + 1] + \
                       obs[joint_vel2.start:joint_vel2.end + 1] + \
                       obs[body_vel.start:body_vel.end + 1] + \
                       obs[body_vel_rot.start:body_vel_rot.end + 1] + \
                       obs[mass_center_vel.start: mass_center_vel.end + 1]
    elif target == "acc":
        mass_center_acc = slice_point(start = 409, end = 411)
        if isinstance(obs, numpy.ndarray):
            if len(obs.shape) == 2:
                new_obs = obs[:, numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
            elif len(obs.shape) == 1:
                new_obs = obs[numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
            else:
                print("Input shape error!", obs.shape)
        elif isinstance(obs, list):
            new_obs = []
            new_obs += obs[mass_center_acc.start:mass_center_acc.end + 1]

    return new_obs

def demo_filter(obs, target="pos"):
    if target == "pos":
        joint_pos = slice_point(start=0, end=13)

        body_pos = slice_point(start=28, end=60)

        body_pos_rot = slice_point(start=94, end=126)

        mass_center_pos = slice_point(start=160, end=162)

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
        joint_vel = slice_point(start=14, end=27)

        body_vel = slice_point(start=61, end=93)

        body_vel_rot = slice_point(start=127, end=159)

        mass_center_vel = slice_point(start=163, end=165)

        if isinstance(obs, numpy.ndarray):
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
            new_obs +=  obs[joint_vel.start:joint_vel.end + 1] + \
                        obs[body_vel.start:body_vel.end + 1] + \
                        obs[body_vel_rot.start:body_vel_rot.end + 1] + \
                        obs[mass_center_vel.start:mass_center_vel.end + 1]
    elif target == "acc":
        mass_center_acc = slice_point(start=166, end=168)
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



def target_filter_v2(obs, target="pos"):
    new_obs = None
    # 412 -> 169
    #joint_pos = slice_point(start=0, end=15)
    joint_pos0 = slice_point(start=0, end=2)
    joint_pos1 = slice_point(start=6, end=7)
    joint_pos2 = slice_point(start=9, end=9)
    joint_pos3 = slice_point(start=11, end=12)
    joint_pos4 = slice_point(start=14, end=15)

    body_pos = slice_point(start = 51, end = 83)

    body_pos_rot = slice_point(start=150, end=182)

    mass_center_pos = slice_point(start=403, end=405)

    if target == "pos":
        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[joint_pos0.start:joint_pos0.end + 1, \
                                 joint_pos1.start:joint_pos1.end + 1, \
                                 joint_pos2.start:joint_pos2.end + 1, \
                                 joint_pos3.start:joint_pos3.end + 1, \
                                 joint_pos4.start:joint_pos4.end + 1, \
                                 body_pos.start:body_pos.end + 1, \
                                 body_pos_rot.start:body_pos_rot.end + 1, \
                                 mass_center_pos.start: mass_center_pos.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[joint_pos0.start:joint_pos0.end + 1, \
                                 joint_pos1.start:joint_pos1.end + 1, \
                                 joint_pos2.start:joint_pos2.end + 1, \
                                 joint_pos3.start:joint_pos3.end + 1, \
                                 joint_pos4.start:joint_pos4.end + 1, \
                                 body_pos.start:body_pos.end + 1, \
                                 body_pos_rot.start:body_pos_rot.end + 1, \
                                 mass_center_pos.start: mass_center_pos.end + 1]]
        else:
            print("Input shape error!", obs.shape)

    elif target == "vel":
        # joint_vel = slice_point(start=17, end=32)
        joint_vel0 = slice_point(start=17, end=19)
        joint_vel1 = slice_point(start=23, end=24)
        joint_vel2 = slice_point(start=26, end=26)
        joint_vel3 = slice_point(start=28, end=29)
        joint_vel4 = slice_point(start=31, end=32)

        body_vel = slice_point(start = 84, end = 116)

        body_vel_rot = slice_point(start=183, end=215)

        mass_center_vel = slice_point(start=406, end=408)

        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[joint_vel0.start:joint_vel0.end + 1, \
                                 joint_vel1.start:joint_vel1.end + 1, \
                                 joint_vel2.start:joint_vel2.end + 1, \
                                 joint_vel3.start:joint_vel3.end + 1, \
                                 joint_vel4.start:joint_vel4.end + 1, \
                                 body_vel.start:body_vel.end + 1, \
                                 body_vel_rot.start:body_vel_rot.end + 1, \
                                 mass_center_vel.start: mass_center_vel.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[joint_vel0.start:joint_vel0.end + 1, \
                                 joint_vel1.start:joint_vel1.end + 1, \
                                 joint_vel2.start:joint_vel2.end + 1, \
                                 joint_vel3.start:joint_vel3.end + 1, \
                                 joint_vel4.start:joint_vel4.end + 1, \
                                 body_vel.start:body_vel.end + 1, \
                                 body_vel_rot.start:body_vel_rot.end + 1, \
                                 mass_center_vel.start: mass_center_vel.end + 1]]
        else:
            print("Input shape error!", obs.shape)

    elif target == "acc":
        mass_center_acc = slice_point(start = 409, end = 411)
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
        else:
            print("Input shape error!", obs.shape)
    elif target == "center_pos":
        mass_center_pos = slice_point(start=403, end=405)
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[mass_center_pos.start:mass_center_pos.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[mass_center_pos.start:mass_center_pos.end + 1]]
        else:
            print("Input shape error!", obs.shape)

    return new_obs

def demo_filter_v2(obs, target="pos"):
    if target == "pos":
        joint_pos0 = slice_point(start=0, end=2)
        joint_pos1 = slice_point(start=6, end=8)
        joint_pos2 = slice_point(start=10, end=13)

        body_pos = slice_point(start=28, end=60)

        body_pos_rot = slice_point(start=94, end=126)

        mass_center_pos = slice_point(start=160, end=162)

        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[joint_pos0.start:joint_pos0.end + 1,
                             joint_pos1.start:joint_pos1.end + 1,
                             joint_pos2.start:joint_pos2.end + 1,
                             body_pos.start:body_pos.end + 1,
                             body_pos_rot.start:body_pos_rot.end + 1, \
                             mass_center_pos.start:mass_center_pos.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[joint_pos0.start:joint_pos0.end + 1,
                             joint_pos1.start:joint_pos1.end + 1,
                             joint_pos2.start:joint_pos2.end + 1,
                             body_pos.start:body_pos.end + 1,
                             body_pos_rot.start:body_pos_rot.end + 1, \
                             mass_center_pos.start:mass_center_pos.end + 1]]
        else:
            print("Input shape error!", obs.shape)
    elif target == "vel":
        joint_vel0 = slice_point(start=14, end=16)
        joint_vel1 = slice_point(start=20, end=22)
        joint_vel2 = slice_point(start=24, end=27)

        body_vel = slice_point(start=61, end=93)

        body_vel_rot = slice_point(start=127, end=159)

        mass_center_vel = slice_point(start=163, end=165)

        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[joint_vel0.start:joint_vel0.end + 1,
                             joint_vel1.start:joint_vel1.end + 1,
                             joint_vel2.start:joint_vel2.end + 1,
                             body_vel.start:body_vel.end + 1,
                             body_vel_rot.start:body_vel_rot.end + 1, \
                             mass_center_vel.start:mass_center_vel.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[joint_vel0.start:joint_vel0.end + 1,
                         joint_vel1.start:joint_vel1.end + 1,
                         joint_vel2.start:joint_vel2.end + 1,
                         body_vel.start:body_vel.end + 1,
                         body_vel_rot.start:body_vel_rot.end + 1, \
                         mass_center_vel.start:mass_center_vel.end + 1]]
        else:
            print("Input shape error!", obs.shape)
    elif target == "acc":
        mass_center_acc = slice_point(start=166, end=168)
        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[mass_center_acc.start:mass_center_acc.end + 1]]
        else:
            print("Input shape error!", obs.shape)
    elif target == "center_pos":
        mass_center_pos = slice_point(start=160, end=162)
        # new_obs = obs[:, numpy.r_[0:33 + 1, 51:116 + 1, 150:215 + 1, 403:411 + 1]]
        if len(obs.shape) == 2:
            new_obs = obs[:, numpy.r_[mass_center_pos.start:mass_center_pos.end + 1]]
        elif len(obs.shape) == 1:
            new_obs = obs[numpy.r_[mass_center_pos.start:mass_center_pos.end + 1]]
        else:
            print("Input shape error!", obs.shape)
    return new_obs