from functools import partial
import supersuit
from pettingzoo import atari

# from pettingzoo.atari import basketball_pong_v3
# from pettingzoo.atari import boxing_v2
# from pettingzoo.atari import combat_tank_v2
# from pettingzoo.atari import double_dunk_v3
# from pettingzoo.atari import entombed_competitive_v3
# from pettingzoo.atari import entombed_cooperative_v3
# from pettingzoo.atari import flag_capture_v2
# from pettingzoo.atari import foozpong_v3
# from pettingzoo.atari import ice_hockey_v2
# from pettingzoo.atari import joust_v3
# from pettingzoo.atari import mario_bros_v3
# from pettingzoo.atari import maze_craze_v3
# from pettingzoo.atari import othello_v3
# from pettingzoo.atari import pong_v3
# from pettingzoo.atari import quadrapong_v4
# from pettingzoo.atari import space_invaders_v2
# from pettingzoo.atari import space_war_v2
# from pettingzoo.atari import surround_v2
# from pettingzoo.atari import tennis_v3
# from pettingzoo.atari import video_checkers_v4
# from pettingzoo.atari import volleyball_pong_v2
# from pettingzoo.atari import warlords_v3
# from pettingzoo.atari import wizard_of_wor_v3

competition_env = [
    "basketball_pong",
    "boxing",
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
    env = env.parallel_env(obs_type='grayscale_image', **kwargs)
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.frame_stack_v1(env, 4)
    env.reset()
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
    'basketball_pong': partial(ma_atari_fn, env=atari.basketball_pong_v3, env_name='basketball_pong'),
    'boxing': partial(ma_atari_fn, env=atari.boxing_v2, env_name='boxing'),
    'combat_tank': partial(ma_atari_fn, env=atari.combat_tank_v2, env_name='combat_tank'),
    'double_dunk': partial(ma_atari_fn, env=atari.double_dunk_v3, env_name='double_dunk'),
    'entombed_competitive': partial(ma_atari_fn, env=atari.entombed_competitive_v3, env_name='entombed_competitive'),
    'entombed_cooperative': partial(ma_atari_fn, env=atari.entombed_cooperative_v3, env_name='entombed_cooperative'),
    'flag_capture': partial(ma_atari_fn, env=atari.flag_capture_v2, env_name='flag_capture'),
    'foozpong': partial(ma_atari_fn, env=atari.foozpong_v3, env_name='foozpong'),
    'ice_hockey': partial(ma_atari_fn, env=atari.ice_hockey_v2, env_name='ice_hockey'),
    'joust': partial(ma_atari_fn, env=atari.joust_v3, env_name='joust'),
    'mario_bros': partial(ma_atari_fn, env=atari.mario_bros_v3, env_name='mario_bros'),
    'maze_craze': partial(ma_atari_fn, env=atari.maze_craze_v3, env_name='maze_craze'),
    'othello': partial(ma_atari_fn, env=atari.othello_v3, env_name='othello'),
    'pong': partial(ma_atari_fn, env=atari.pong_v3, env_name='pong'),
    'quadrapong': partial(ma_atari_fn, env=atari.quadrapong_v4, env_name='quadrapong'),
    'space_invaders': partial(ma_atari_fn, env=atari.othello_v3, env_name='space_invaders'),
    'space_war': partial(ma_atari_fn, env=atari.space_war_v2, env_name='space_war'),
    'surround': partial(ma_atari_fn, env=atari.surround_v2, env_name='surround'),
    'tennis': partial(ma_atari_fn, env=atari.tennis_v3, env_name='tennis'),
    'video_checkers': partial(ma_atari_fn, env=atari.video_checkers_v4, env_name='video_checkers'),
    'volleyball_pong': partial(ma_atari_fn, env=atari.volleyball_pong_v2, env_name='volleyball_pong'),
    'warlords': partial(ma_atari_fn, env=atari.warlords_v3, env_name='warlords'),
    'wizard_of_wor': partial(ma_atari_fn, env=atari.wizard_of_wor_v3, env_name='wizard_of_wor'),
}
