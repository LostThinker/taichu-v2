import pettingzoo as pz
import numpy as np
import supersuit


def ma_atari_test():
    from pettingzoo.atari import basketball_pong_v3
    env = basketball_pong_v3.parallel_env(num_players=2, obs_type='grayscale_image')
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.frame_stack_v1(env, 4)
    returns=env.reset()
    action_dict={}
    for agent in env.agents:
        print(agent)
        obs_s=env.observation_space(agent)
        action = env.action_space(agent).sample()
        action_dict[agent]=action
    returns=env.step(action_dict)
    print(returns)


def sc2env_test():
    from smac.env.starcraft2.starcraft2 import StarCraft2Env
    env = StarCraft2Env()
    env.reset()
    actions = []
    for agent_id in range(env.n_agents):
        avail_actions = env.get_avail_agent_actions(agent_id)
        avail_actions_ind = np.nonzero(avail_actions)[0]
        action = np.random.choice(avail_actions_ind)
        actions.append(action)
    reward, terminated, info = env.step(actions)
    print(info)

    # for agent_id, agent_info in env.agents.items():
    #     if agent_info.unit_type == env.marine_id:
    #         agent_type = "marine"
    #     elif agent_info.unit_type == env.marauder_id:
    #         agent_type = "marauder"
    #     elif agent_info.unit_type == env.medivac_id:
    #         agent_type = "medivac"
    #     elif agent_info.unit_type == env.hydralisk_id:
    #         agent_type = "hydralisk"
    #     elif agent_info.unit_type == env.zergling_id:
    #         agent_type = "zergling"
    #     elif agent_info.unit_type == env.baneling_id:
    #         agent_type = "baneling"
    #     elif agent_info.unit_type == env.stalker_id:
    #         agent_type = "stalker"
    #     elif agent_info.unit_type == env.colossus_id:
    #         agent_type = "colossus"
    #     elif agent_info.unit_type == env.zealot_id:
    #         agent_type = "zealot"
    #     else:
    #         raise AssertionError(f"agent type {agent_type} not supported")


def pz_sc2_test():
    from smac.env.pettingzoo import StarCraft2PZEnv as sc2
    env = sc2.env(map_name="corridor")
    print(env.agents)


# def pz_env_test():
#     from pettingzoo.butterfly import knights_archers_zombies_v10
#     env = knights_archers_zombies_v10.env()
#     env.reset()
#     for agent in env.agent_iter():
#         observation, reward, done, info = env.last()
#         action = policy(observation, agent)
#         env.step(action)
if __name__ == '__main__':
    ma_atari_test()
