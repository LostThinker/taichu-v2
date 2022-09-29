from smac.env.starcraft2.starcraft2 import StarCraft2Env
from gym import spaces
import numpy as np
from abc import ABC, abstractmethod
from pettingzoo import atari
import supersuit
from ma_atari_env import MaAtariEnv_register


class GroupBaseEnv(ABC):
    def __init__(self, game_name, group_type='unit_type'):
        self.game_name = game_name
        self.group_type = group_type
        self.camp = {}
        self.camp_num = 0
        self.group_num = 0

        self.units_info = dict(
            unit_camp_id_map={},
            unit_camp_list=[],
            unit_type_id_map={},
            unit_type_list=[],  # 单位的种类
            unit_action_space_type_list=[],  # action space的种类
            unit_obs_space_type_list=[],  # obs space的种类
            unit_action_space_list=[],  # 每一个unit的action space
            unit_obs_space_list=[]  # 每一个unit的obs space
        )

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action_dict):
        pass

    @abstractmethod
    def _init_units(self):
        pass

    def _group_units(self):
        self.camp = {}
        if self.group_type == "unit_type":
            for unit_camp_id, unit_camp in enumerate(self.units_info["unit_camp_list"]):
                group = {}
                for unit_type_id, unit_type in enumerate(self.units_info["unit_type_list"]):
                    group[unit_type_id] = self.units_info["unit_type_id_map"][unit_type]
                self.camp[unit_camp_id] = group
        elif self.group_type == "unit_space":
            for unit_camp_id, unit_camp in enumerate(self.units_info["unit_camp_list"]):
                group = {}
                action_obs_space_pair_list = []
                for i in range(self.unit_num):
                    action_obs_space_pair = (
                        self.units_info["unit_action_space_list"], self.units_info["unit_obs_space_list"])
                    if action_obs_space_pair not in action_obs_space_pair_list:
                        action_obs_space_pair_list.append(action_obs_space_pair)
                        unit_type_id = action_obs_space_pair_list.index(action_obs_space_pair)
                        group[unit_type_id] = [i]
                    else:
                        unit_type_id = action_obs_space_pair_list.index(action_obs_space_pair)
                        group[unit_type_id].append(i)
                self.camp[unit_camp_id] = group
        else:
            raise NotImplementedError
        self.camp_num = len(self.camp)
        for camp in self.camp.values():
            self.group_num += len(camp)

    @abstractmethod
    def _update_alive_units(self):
        pass


class GroupSC2Env(GroupBaseEnv):
    def __init__(
            self,
            game_name='1c3s5z',
            group_type='unit_type'

    ):
        super().__init__(
            game_name=game_name,
            group_type=group_type
        )
        self.env = StarCraft2Env(map_name=game_name)
        self.env.reset()
        self.action_dim = self.env.n_actions
        self.unit_num = self.env.n_agents
        self.alive_unit_id_list = [i for i in range(self.unit_num)]
        self.unit_dead_flag = [0 for _ in range(self.unit_num)]

        self._init_units()
        self._group_units()

    def reset(self):
        obs, state = self.env.reset()
        self.alive_unit_id_list = [i for i in range(self.unit_num)]
        obs = np.array(obs)
        action_mask = np.array(self.env.get_avail_actions())
        obs_dict = {}
        camp_shared_info_dict = {}

        for camp_id, group in self.camp.items():
            obs_dict[camp_id] = {}
            for group_id, units_id_list in group.items():
                obs_dict[camp_id][group_id] = {}
                for unit_id in units_id_list:
                    # key
                    obs_dict[camp_id][group_id][unit_id] = dict(
                        unit_id=unit_id,  #
                        obs=obs[unit_id, :],
                        action_mask=action_mask[unit_id, :],
                    )
            # key
            camp_shared_info_dict[camp_id] = dict(
                global_state=state,
            )
        return obs_dict, camp_shared_info_dict

    def step(self, action_dict):
        game_input_actions = [None for _ in range(self.unit_num)]
        noop_action = np.zeros(self.action_dim)
        for camp_id, camp_action in action_dict.items():
            for group_id, group_action in camp_action.items():
                for unit_id, unit_action in group_action.items():
                    game_input_actions[unit_id] = unit_action
        for i in range(len(game_input_actions)):
            if game_input_actions[i] is None:
                game_input_actions[i] = noop_action

        reward, done, info = self.env.step(game_input_actions)
        self._update_alive_units()
        next_obs = self.env.get_obs()
        next_state = self.env.get_state()
        action_mask = np.array(self.env.get_avail_actions())

        reward_dict = {}
        done_dict = {}
        next_obs_dict = {}
        info_dict = {}
        camp_shared_info_dict = {}

        for camp_id, camp_action in action_dict.items():
            reward_dict[camp_id] = {}
            done_dict[camp_id] = {}
            next_obs_dict[camp_id] = {}
            info_dict[camp_id] = {}
            for group_id, group_action in camp_action.items():
                reward_dict[camp_id][group_id] = {}
                done_dict[camp_id][group_id] = {}
                next_obs_dict[camp_id][group_id] = {}
                info_dict[camp_id][group_id] = {}
                for unit_id, unit_action in group_action.items():
                    if unit_id in self.alive_unit_id_list or self.unit_dead_flag[unit_id]:
                        # 保证智能体死亡时的数据传输，死亡后的数据就截断
                        next_obs_dict[camp_id][group_id][unit_id] = dict(
                            unit_id=unit_id,  #
                            obs=next_obs[unit_id],
                            action_mask=action_mask[unit_id],
                        )
                        reward_dict[camp_id][group_id][unit_id] = reward
                        done_dict[camp_id][group_id][unit_id] = done or (
                            True if self.unit_dead_flag[unit_id] else False)
                        info_dict[camp_id][group_id][unit_id] = None

            camp_shared_info_dict[camp_id] = dict(
                global_state=next_state,
                game_finished=done,
                game_info=info
            )

        return next_obs_dict, reward_dict, done_dict, info_dict, camp_shared_info_dict

    def _init_units(self):
        for unit_id, unit_info in self.env.agents.items():
            unit_action_space = spaces.Discrete(
                self.env.get_total_actions() - 1
            )  # no-op in dead units is not an action
            unit_obs_space = spaces.Box(float("-inf"), float("inf"), shape=(self.env.get_obs_size(),))
            unit_camp = 'cooper_camp'
            if unit_info.unit_type == self.env.marine_id:
                unit_type = "marine"
            elif unit_info.unit_type == self.env.marauder_id:
                unit_type = "marauder"
            elif unit_info.unit_type == self.env.medivac_id:
                unit_type = "medivac"
            elif unit_info.unit_type == self.env.hydralisk_id:
                unit_type = "hydralisk"
            elif unit_info.unit_type == self.env.zergling_id:
                unit_type = "zergling"
            elif unit_info.unit_type == self.env.baneling_id:
                unit_type = "baneling"
            elif unit_info.unit_type == self.env.stalker_id:
                unit_type = "stalker"
            elif unit_info.unit_type == self.env.colossus_id:
                unit_type = "colossus"
            elif unit_info.unit_type == self.env.zealot_id:
                unit_type = "zealot"
            else:
                raise AssertionError(f"unit type {unit_type} not supported")

            if unit_camp not in self.units_info['unit_camp_list']:
                self.units_info["unit_camp_list"].append(unit_camp)
            if unit_type not in self.units_info['unit_type_list']:
                self.units_info["unit_type_list"].append(unit_type)

            if unit_action_space not in self.units_info['unit_action_space_type_list']:
                self.units_info['unit_action_space_type_list'].append(unit_action_space)
            if unit_obs_space not in self.units_info['unit_obs_space_type_list']:
                self.units_info['unit_obs_space_type_list'].append(unit_obs_space)

            if self.units_info["unit_camp_id_map"].get(unit_camp) is None:
                self.units_info["unit_camp_id_map"][unit_camp] = [unit_id]
            else:
                self.units_info["unit_camp_id_map"][unit_camp].append(unit_id)

            if self.units_info["unit_type_id_map"].get(unit_type) is None:
                self.units_info["unit_type_id_map"][unit_type] = [unit_id]
            else:
                self.units_info["unit_type_id_map"][unit_type].append(unit_id)

            self.units_info["unit_action_space_list"].append(unit_action_space)
            self.units_info["unit_obs_space_list"].append(unit_obs_space)

    def _update_alive_units(self):
        self.unit_dead_flag = [0 for _ in range(self.unit_num)]
        for unit_id, unit in self.env.agents.items():
            if unit.health == 0:
                if unit_id in self.alive_unit_id_list:
                    self.unit_dead_flag[unit_id] = 1
                    self.alive_unit_id_list.remove(unit_id)


class GroupMaAtariEnv(GroupBaseEnv):
    def __init__(
            self,
            game_name='basketball_pong',
            group_type='unit_type'

    ):
        super().__init__(
            game_name=game_name,
            group_type=group_type
        )
        self.env, self.env_type = MaAtariEnv_register[game_name]
        self.unit_num = self.env.num_agents
        self.alive_unit_id_list = [i for i in range(self.unit_num)]
        self.unit_dead_flag = [0 for _ in range(self.unit_num)]
        self.unit_name_list = self.env.possible_agents

        self._init_units()
        self._group_units()

    def reset(self):
        obs_dict = {}
        camp_shared_info_dict = {}

        raw_obs = self.env.reset()
        for camp_id, group in self.camp.items():
            obs_dict[camp_id] = {}
            for group_id, units_id_list in group.items():
                obs_dict[camp_id][group_id] = {}
                for unit_id in units_id_list:
                    unit_name = self.unit_name_list[unit_id]
                    obs_dict[camp_id][group_id][unit_id] = dict(
                        unit_id=unit_id,  #
                        obs=raw_obs[unit_name],
                    )
            camp_shared_info_dict[camp_id] = {}
        return obs_dict, camp_shared_info_dict

    def step(self, action_dict):
        game_input_actions = {}
        for camp_id, camp_action in action_dict.items():
            for group_id, group_action in camp_action.items():
                for unit_id, unit_action in group_action.items():
                    unit_name = self.unit_name_list[unit_id]
                    game_input_actions[unit_name] = unit_action

        for unit_name in self.env.agents:
            if game_input_actions.get(unit_name) is None:
                noop_action = self.env.action_space(unit_name).sample()
                game_input_actions[unit_name] = noop_action

        next_obs, reward, done, truncate, info = self.env.step(game_input_actions)
        self._update_alive_units()

        reward_dict = {}
        done_dict = {}
        next_obs_dict = {}
        info_dict = {}
        camp_shared_info_dict = {}

        for camp_id, group in self.camp.items():
            reward_dict[camp_id] = {}
            done_dict[camp_id] = {}
            next_obs_dict[camp_id] = {}
            info_dict[camp_id] = {}
            for group_id, units_id_list in group.items():
                reward_dict[camp_id][group_id] = {}
                done_dict[camp_id][group_id] = {}
                next_obs_dict[camp_id][group_id] = {}
                info_dict[camp_id][group_id] = {}
                for unit_id in units_id_list:
                    unit_name = self.unit_name_list[unit_id]
                    unit_next_obs = next_obs.get(unit_name)
                    if unit_next_obs is not None:
                        next_obs_dict[camp_id][group_id][unit_id] = dict(
                            unit_id=unit_id,
                            obs=unit_next_obs,
                        )
                        reward_dict[camp_id][group_id][unit_id] = reward[unit_name]
                        done_dict[camp_id][group_id][unit_id] = done[unit_name]
                        info_dict[camp_id][group_id][unit_id] = info[unit_name]
            camp_shared_info_dict[camp_id] = {}

        # for camp_id, camp_action in action_dict.items():
        #     reward_dict[camp_id] = {}
        #     done_dict[camp_id] = {}
        #     next_obs_dict[camp_id] = {}
        #     info_dict[camp_id] = {}
        #     for group_id, group_action in camp_action.items():
        #         reward_dict[camp_id][group_id] = {}
        #         done_dict[camp_id][group_id] = {}
        #         next_obs_dict[camp_id][group_id] = {}
        #         info_dict[camp_id][group_id] = {}
        #         for unit_id in group_action.keys():
        #             unit_name = self.unit_name_list[unit_id]

    def _init_units(self):
        self.units_info = dict(
            unit_camp_id_map={},
            unit_camp_list=[],
            unit_type_id_map={},
            unit_type_list=[],  # 单位的种类
            unit_action_space_type_list=[],  # action space的种类
            unit_obs_space_type_list=[],  # obs space的种类
            unit_action_space_list=[],  # 每一个unit的action space
            unit_obs_space_list=[]  # 每一个unit的obs space
        )

        for unit_name in self.env.agents:
            unit_type = unit_name
            if self.env_type == 'group_compet_env':
                if unit_name in ['first_0', 'third_0']:
                    unit_camp = 'camp1'
                else:
                    unit_camp = 'camp2'
            elif self.env_type in ['compet_env', "mix_env"]:
                unit_camp = unit_name
            elif self.env_type == "cooper_env":
                unit_camp = "cooper_camp"
            else:
                raise ValueError

            unit_id = self.unit_name_list.index(unit_name)
            unit_action_space = self.env.action_space(unit_name)
            unit_obs_space = self.env.observation_space(unit_name)
            if unit_camp not in self.units_info['unit_camp_list']:
                self.units_info["unit_camp_list"].append(unit_camp)
            if unit_type not in self.units_info['unit_type_list']:
                self.units_info["unit_type_list"].append(unit_type)

            if unit_action_space not in self.units_info['unit_action_space_type_list']:
                self.units_info['unit_action_space_type_list'].append(unit_action_space)
            if unit_obs_space not in self.units_info['unit_obs_space_type_list']:
                self.units_info['unit_obs_space_type_list'].append(unit_obs_space)

            if self.units_info["unit_camp_id_map"].get(unit_camp) is None:
                self.units_info["unit_camp_id_map"][unit_camp] = [unit_id]
            else:
                self.units_info["unit_camp_id_map"][unit_camp].append(unit_id)

            if self.units_info["unit_type_id_map"].get(unit_type) is None:
                self.units_info["unit_type_id_map"][unit_type] = [unit_id]
            else:
                self.units_info["unit_type_id_map"][unit_type].append(unit_id)

            self.units_info["unit_action_space_list"].append(unit_action_space)
            self.units_info["unit_obs_space_list"].append(unit_obs_space)

    def _group_units(self):
        if self.group_type == "unit_type":
            for unit_camp_id, unit_camp in enumerate(self.units_info["unit_camp_list"]):
                group = {}
                for unit_type_id, unit_type in enumerate(self.units_info["unit_type_list"]):
                    group[unit_type_id] = self.units_info["unit_type_id_map"][unit_type]
                self.camp[unit_camp_id] = group
        elif self.group_type == "unit_space":
            for unit_camp_id, unit_camp in enumerate(self.units_info["unit_camp_list"]):
                group = {}
                action_obs_space_pair_list = []
                for i in range(self.unit_num):
                    action_obs_space_pair = (
                        self.units_info["unit_action_space_list"], self.units_info["unit_obs_space_list"])
                    if action_obs_space_pair not in action_obs_space_pair_list:
                        action_obs_space_pair_list.append(action_obs_space_pair)
                        unit_type_id = action_obs_space_pair_list.index(action_obs_space_pair)
                        group[unit_type_id] = [i]
                    else:
                        unit_type_id = action_obs_space_pair_list.index(action_obs_space_pair)
                        group[unit_type_id].append(i)
                self.camp[unit_camp_id] = group
        else:
            raise NotImplementedError
        self.camp_num = len(self.camp)
        for camp in self.camp.values():
            self.group_num += len(camp)

    def _update_alive_units(self):
        pass


if __name__ == '__main__':
    env = GroupSC2Env(group_type="unit_space")
    obs, state = env.reset()
    while True:
        action_dict = {}
        for camp_id, group in env.camp.items():
            action_dict[camp_id] = {}
            for group_id, units_id_list in group.items():
                action_dict[camp_id][group_id] = {}
                for unit_id in units_id_list:
                    avail_actions = env.env.get_avail_agent_actions(unit_id)
                    avail_actions_ind = np.nonzero(avail_actions)[0]
                    action = np.random.choice(avail_actions_ind)
                    action_dict[camp_id][group_id][unit_id] = action
        next_obs_dict, reward_dict, done_dict, info_dict, camp_shared_info_dict = env.step(action_dict)
        if camp_shared_info_dict[0]['game_finished']:
            break
    print(env.camp)
    print(env.camp_num)
    print(env.group_num)
