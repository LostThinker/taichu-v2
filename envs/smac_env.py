from smac.smac.env.starcraft2.starcraft2 import StarCraft2Env
import numpy as np
from utils.utils import get_obs


class SC2Env(object):
    """
    标准化的环境接口，返回值均为(n_agents,dim)形状,除了global_state为(dim,)
    """

    def __init__(self, map_name='8m', difficulty='7', seed=0, max_step=500):
        self.env = StarCraft2Env(map_name=map_name, difficulty=difficulty, seed=seed)
        self.env_info = self.env.get_env_info()
        self.n_agents = self.env_info['n_agents']
        self.n_actions = self.env_info['n_actions']
        self.total_step = 0

    def reset(self):
        agent_obs, state = self.env.reset()
        global_state, last_action = self._process_state(state)

        agent_obs = np.array(agent_obs)
        action_mask = np.array(self.env.get_avail_actions())
        agent_id = np.eye(self.n_agents, dtype=np.float32)

        time_step_obs = get_obs(agent_obs, last_action, global_state, action_mask, agent_id)

        return time_step_obs

    def step(self, actions):
        for a_id, action in enumerate(actions):
            avail_actions = self.env.get_avail_agent_actions(a_id)
            if avail_actions[action] != 1:
                print(actions)
        reward, done, info = self.env.step(actions)
        if info == {}
            info = dict(
                env_type='SC2',
                env_state='abnormal',
                battle_won=False,
                dead_allies=self.n_agents,
                dead_enemies=0
            )
        else:
            info = dict(
                env_type='SC2',
                env_state='nomal',
                battle_won=info['battle_won'],
                dead_allies=info['dead_enemies']
            )

        self.total_step += 1
        agent_obs = self.env.get_obs()
        state = self.env.get_state()

        global_state, last_action = self._process_state(state)
        agent_obs = np.array(agent_obs)
        action_mask = np.array(self.env.get_avail_actions())
        agent_id = np.eye(self.n_agents, dtype=np.float32)

        next_obs = get_obs(agent_obs, last_action, global_state, action_mask, agent_id)

        return next_obs, reward, done, info

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = dict(
            state_shape=self.env_info['state_shape'] - self.n_agents * self.n_actions,
            obs_shape=self.env_info['obs_shape'],
            n_action=self.env_info['n_actions'],
            n_agent=self.env_info['n_agents'],
            episode_limit=self.env_info['episode_limit']
        )
        return env_info

    def _process_state(self, state):
        global_state = state[:-self.n_agents * self.n_actions]
        last_action = state[-self.n_agents * self.n_actions:]
        last_action = np.array(last_action).reshape(self.n_agents, self.n_actions)
        global_state = np.array(global_state)
        return global_state, last_action
