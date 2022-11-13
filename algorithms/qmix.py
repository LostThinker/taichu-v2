from utils.utils import dictToObj, get_env_config, make_policy, dict_state_to_tensor, batch_data_processor, \
    tensor_to_dict_state, prev_state_split, get_optimizer, target_net_update
from .base_algo import BaseAlgo
import torch
import torch.nn.functional as F
import copy
from torch import nn


#####################config#######################
def qmix_default_config():
    policy_config = dict(
        cuda=True,
        batch_size=64,
        update_per_collect=1,
        learning_rate=0.0005,
        optimizer='adam',

        target_update_theta=1,
        target_update_step=200,
        gamma=0.99,
    )

    env_config = dict(
        env_type='SC2',
        env_args=dict(
            map_name="8m",
            difficulty="7",
            seed=0,
            max_step=500
        )  # 传入环境里的具体初始化参数
    )
    env_config = get_env_config(env_config)

    collect_config = dict(
        collector_num=3,
        collector_address_list=[('127.0.0.1', 50000), ('127.0.0.1', 51000), ('127.0.0.1', 52000)],
        sample_type='unroll',
        n_sample=60,
        unroll_len=20,
        vec_env_config=dict(
            env_num=6,
            env_fn=env_config['env_fn'],
            env_args=env_config['env_args'],
            max_step=500,
        ),
        epsilon_start=1,
        epsilon_end=0.02,
        epsilon_step=100000,
        max_collect_step=2000000,  # 结束条件之一，设为None表示忽略该条件,可以多个条件并存，优先触发
        update_params_period=3  # 用于并行，每隔多少秒更新参数

    )

    eval_config = dict(
        eval_address=('127.0.0.1', 40000),
        eval_episode_num=20,
        vec_env_config=dict(
            env_num=2,
            env_fn=env_config['env_fn'],
            env_args=env_config['env_args'],
            max_step=500,
        ),
        eval_interval=2000,

        eval_time_period=10,  # 用于并行，每隔几秒eval一次
        end_reward_condition=None,  # 结束条件之一，设为None表示忽略该条件,可以多个条件并存，优先触发
        repeat_eval_times=5,  # 连续n次eval的平均奖励都达到end_reward_condition则表示收敛,默认5

    )

    model_config = dict(
        agent_num=env_config['env_info']['n_agent'],
        obs_shape=env_config['env_info']['obs_shape'],
        action_shape=env_config['env_info']['n_action'],
        state_shape=env_config['env_info']['state_shape'],
        actor_hidden_size_list=[256, 256, 512],
        hidden_channel_list=[16, 32, 32],
        conv_output_shape=128,
        kernel_size_list=[8, 4, 3],
        stride_list=[4, 2, 1],
        activation='relu',
        actor_rnn_type='gru',
        actor_rnn_hidden_size=256,
        mixing_embed_shape=32,
        hypernet_embed=64,
        norm_type=None,
        noise=False
    )

    learner_config = dict(
        learner_address=('127.0.0.1', 30000),
        train_epoch=100000,  # 结束条件之一，设为None表示忽略该条件,可以多个条件并存，优先触发
        update_per_collect=1,
        batch_size=64,
        pause_time=5,  # 用于控制生成消耗比
    )

    replaybuffer_config = dict(
        replaybuffer_address=('127.0.0.1', 20000),
        args=dict(
            size=5000,
        )
    )

    config = dict(
        algorithm='qmix',
        env_config=env_config,
        model_config=model_config,
        policy_config=policy_config,
        collect_config=collect_config,
        learner_config=learner_config,
        eval_config=eval_config,
        replaybuffer_config=replaybuffer_config
    )

    return dictToObj(config)


class QMIXPolicy():
    def __init__(self, policy_config, model):
        self.policy_config = policy_config
        self.model = model
        self.optimizer = get_optimizer(policy_config.optimizer)(self.model.parameters(), lr=self.policy_config.learning_rate)
        self.target_model = copy.deepcopy(self.model)
        self.hidden_state = None
        self.total_train_step = 0
        if policy_config.cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.target_model.to(self.device)

    def inference(self, data):
        self.model.eval()
        data = self.preprocess_inference_data(data)
        T, B, A = data['agent_obs'].shape[:3]
        data = {'obs': data}
        if 'prev_state' not in data.keys:
            data['prev_state'] = dict_state_to_tensor(self.hidden_state, self.device)
        else:
            data['prev_state'] = dict_state_to_tensor(data['prev_state'], self.device)
        output = self.model(data, mode="actor_forward")
        logit, prev_state, next_state, rnn_type = output['logit'], output['prev_state'], output['next_state'], output[
            'rnn_type']
        self.hidden_state = tensor_to_dict_state(next_state, rnn_type, B, A, self.device)  # 此处为内部参数，依旧存在device中
        if 'action_mask' in data['obs'].keys():
            output['action_mask'] = data['obs']['action_mask'].to('cpu')  # output的Tensor都转到cpu
        output.pop('next_state')
        output.pop('rnn_type')
        output['prev_state'] = tensor_to_dict_state(prev_state, rnn_type, B, A)  # output的Tensor都转到cpu
        output['specific'] = [None] * B
        return output

    def train(self, data):
        self.model.train()
        data = self.preprocess_train_data(data)
        prev_state = data['prev_state']  # (T,*) 整个unroll每一步的prev_state
        data['prev_state'] = prev_state[0]  # 用第一个prev_state
        q_total = self.model(data, mode='critic_forward')['q_total']

        next_inputs = {}
        next_inputs['obs'] = data['next_obs']
        next_inputs['prev_state'] = prev_state[1]  # 第二个prev_state
        # 这里可以直接使用next_inputs得到target_q_total，为稳定使用double q
        next_action = self.model(next_inputs, mode='actor_forward')['logit'].clone().detach().argmax(
            dim=-1)
        next_inputs['action'] = next_action

        with torch.no_grad():
            target_q_total = self.target_model(next_inputs, mode='critic_forward')['q_total']
            target_y = self.policy_config.gamma * (1 - data['done']) * target_q_total + data['reward']

        q_total_loss = F.mse_loss(target_y, q_total)

        self.optimizer.zero_grad()
        q_total_loss.backward()
        self.optimizer.step()
        self.total_train_step += 1

        if self.total_train_step % self.policy_config.target_update_step == 0:
            self.target_model = target_net_update(self.target_model, self.model, 1)

        learn_info = {
            'info_type': 'learn_info',
            'learn_info': {
                'q_total_loss': q_total_loss.item(),
            },
            'total_train_step': self.total_train_step
        }

        return learn_info

    def reset_state(self, env_id):
        if self.hidden_state[env_id]['rnn_type'] == 'lstm':
            self.hidden_state[env_id]['lstm_c'] = torch.zeros_like(torch.tensor(self.hidden_state[env_id]['lstm_c']))
            self.hidden_state[env_id]['lstm_h'] = torch.zeros_like(torch.tensor(self.hidden_state[env_id]['lstm_h']))
        else:
            self.hidden_state[env_id]['hidden_state'] = torch.zeros_like(
                torch.tensor(self.hidden_state[env_id]['hidden_state'])
            )

    def reset(self):
        self.hidden_state = None

    def preprocess_inference_data(self, data):
        data = batch_data_processor(list(data.values()))
        for key in data.keys():
            data[key] = data[key].unsqueeze(0).to(self.device)  # 加入T维度
        return data

    def preprocess_train_data(self, data):
        data = batch_data_processor(data, device=self.device)
        data['prev_state'] = prev_state_split(data['prev_state'])
        data['weight'] = data.get('weight', None)
        data['done'] = data['done'].float()
        return data


class Mixer(nn.module):
    def __int__(
            self,
            agent_num: int,
            state_shape: int,
            mixing_embed_shape: int,
            hypernet_embed: int
    ):
        super(Mixer, self).__init__()
        self.agent_num = agent_num
        self.state_shape = state_shape
        self.mixing_embed_shape = mixing_embed_shape
        self.hypernet_embed = hypernet_embed
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_shape, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, self.mixing_embed_shape * self.agent_num)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_shape, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, self.mixing_embed_shape)
        )

        self.hyper_b_1 = nn.Linear(self.state_shape, self.mixing_embed_shape)

        self.V = nn.Sequential(nn.Linear(self.state_shape, self.mixing_embed_shape), nn.ReLU(),
                               nn.Linear(self.mixing_embed_shape, 1))

    def forward(self, agent_qs, states):
        """
        Overview:
            forward computation graph of pymarl mixer network
        Arguments:
            - agent_qs (:obj:`torch.FloatTensor`): the independent q_value of each agent
            - states (:obj:`torch.FloatTensor`): the emdedding vector of global state
        Returns:
            - q_tot (:obj:`torch.FloatTensor`): the total mixed q_value
        Shapes:
            - agent_qs (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is agent_num
            - states (:obj:`torch.FloatTensor`): :math:`(B, M)`, where M is embedding_size
            - q_tot (:obj:`torch.FloatTensor`): :math:`(B, )`
        """
        bs = agent_qs.shape[:-1]
        states = states.reshape(-1, self.state_shape)
        agent_qs = agent_qs.view(-1, 1, self.agent_num)
        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.agent_num, self.mixing_embed_shape)
        b1 = b1.view(-1, 1, self.mixing_embed_shape)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.mixing_embed_shape, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(*bs)
        return q_tot



