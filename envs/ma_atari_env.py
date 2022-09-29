from .EnvWrapper import GroupBaseEnv
from functools import partial
import supersuit
from pettingzoo.atari import basketball_pong_v3
from pettingzoo.atari import boxing_v2
from pettingzoo.atari import combat_jet_v1
from pettingzoo.atari import combat_tank_v3
from pettingzoo.atari import double_dunk_v3
from pettingzoo.atari import entombed_competitive_v3
from pettingzoo.atari import entombed_cooperative_v3
from pettingzoo.atari import flag_capture_v2
from pettingzoo.atari import foozpong_v3
from pettingzoo.atari import ice_hockey_v2
from pettingzoo.atari import joust_v3
from pettingzoo.atari import mario_bros_v3
from pettingzoo.atari import maze_craze_v3
from pettingzoo.atari import othello_v3
from pettingzoo.atari import pong_v3
from pettingzoo.atari import quadrapong_v4
from pettingzoo.atari import space_invaders_v2
from pettingzoo.atari import space_war_v2
from pettingzoo.atari import surround_v2
from pettingzoo.atari import tennis_v3
from pettingzoo.atari import video_checkers_v4
from pettingzoo.atari import volleyball_pong_v2
from pettingzoo.atari import warlords_v3
from pettingzoo.atari import wizard_of_wor_v3

competition_env = [
    "basketball_pong",
    "boxing",
    "combat_jet",
    "combat_tank",
    "entombed_competitive",
    "flag_capture",
    "maze_craze",
    "othello",
    "pong",
    "space_war",
    "surround"
    "tennis",
    "video_checkers",
    "warlords",
    "wizard_of_wor"

]

cooperation_env = [
    "double_dunk",
    "entombed_cooperative",
    "ice_hockey"
]

group_competition_env = [
    "foozpong",
    "quadrapong",
    "volleyball_pong",

]

mix_env = [
    "mario_bros",
    "space_invaders",
]


def ma_atari_fn(env, env_name, **kwargs):
    env.parallel_env(**kwargs)
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.frame_stack_v1(env, 4)
    if env_name in competition_env:
        env_type = "compet_env"
    elif env_name in cooperation_env:
        env_type = "cooper_env"
    elif env_name in group_competition_env:
        env_type = "group_compet_env"
    elif env_name in mix_env:
        env_type = "mix_env"
    else:
        raise ValueError
    return env, env_type


MaAtariEnv_register = {
    'basketball_pong': partial(ma_atari_fn, env=basketball_pong_v3, env_name='basketball_pong'),
    'foozpong': partial(ma_atari_fn, env=foozpong_v3, env_name='foozpong'),
}
