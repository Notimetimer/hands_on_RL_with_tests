import numpy as np
from math import *
from numpy.linalg import norm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Math_calculates.sub_of_angles import *


def decision_rule(ego_pos_, ego_psi, enm_pos_, distance, ally_missiles, enm_missiles):
    # 输出为所需的绝对高度、相对方位和绝对速度
    # 是否有导弹可用
    has_missile_in_the_air = 0
    for missile in ally_missiles:
        if not missile.dead:
            has_missile_in_the_air = 1
            break
    # 是否被敌机导弹锁定（存在敌机导弹而且距离在20km内，此时导弹雷达可以锁定）
    should_escape = 0
    rel_psi_m = None
    for missile in enm_missiles:
        if not missile.dead:
            line_m_ = missile.pos_ - ego_pos_
            dist_mt = norm(line_m_)
            if dist_mt <= 20e3:  # 假设为导弹雷达的锁定距离
                should_escape = 1
                rel_psi_m = sub_of_radian(atan2(line_m_[2], line_m_[0]), ego_psi)
                break  # trick：先发射的导弹被当做距离最近的，没有在所有导弹中判断距离 todo 判断距离

    action_n = [[0.3, 0.0, 0.0]]

    L_ = enm_pos_ - ego_pos_
    beta = atan2(L_[2], L_[0])
    delta_psi = sub_of_radian(beta, ego_psi)

    # 1 追踪
    if distance > 40e3:  # test
        action_n[0][1] = delta_psi

    # 2 crank
    elif not should_escape and has_missile_in_the_air:
        # print('crank')
        if 0 <= delta_psi:
            temp = 0.4 * (delta_psi - pi / 4) / (pi / 4) * 2
            action_n[0][1] = np.clip(temp, -0.4, 0.4)
        if delta_psi < 0:
            temp = 0.4 * (delta_psi + pi / 4) / (pi / 4) * 2
            action_n[0][1] = np.clip(temp, -0.4, 0.4)
        action_n[0][2] = (1.1 * 340 - 170) / (540 - 170) * 2 - 1  # 速度往1.1Ma去

    # 3 escape
    elif should_escape:
        # print('逃')
        # 水平置尾机动
        temp = 0.4 * sub_of_radian(rel_psi_m, pi) / pi * 4
        action_n[0][1] = np.clip(temp, -0.4, 0.4)
        action_n[0][2] = (1.5 * 340 - 170) / (540 - 170) * 2 - 1  # 速度往1.5Ma去

    # 4 attack again
    else:
        # print('回击')
        action_n[0][1] = delta_psi
        action_n[0][2] = (1.2 * 340 - 170) / (540 - 170) * 2 - 1
    return action_n
