from utils.utils import dictToObj, get_env_config, make_policy, dict_state_to_tensor, batch_data_processor
from .base_algo import BaseAlgo
import torch
import copy


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


class QMIXPolicy(BaseAlgo):
    def __init__(self, env_config, policy_config, model):
        self.env_config = env_config
        self.policy_config = policy_config
        self.model = model
        self.policy = make_policy(env_config, policy_config)
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
        self.target_model.eval()
        data = self.preprocess_inference_data(data)
        T, B, A = data['agent_obs'].shape[:3]
        data = {'obs': data}
        if 'prev_state' not in data.keys:
            data['prev_state'] = dict_state_to_tensor(self.hidden_state, self.device)
        else:
            data['prev_state'] = dict_state_to_tensor(data['prev_state'], self.device)
        output = self.target_model(data, mode="actor_forward")
        logit, prev_state, next_state, rnn_type = output['logit'], output['prev_state'], output['next_state'], output[
            'rnn_type']
        self.hidden_state = ten

    def preprocess_inference_data(self, data):
        data = batch_data_processor(list(data.values()))
        for key in data.keys():
            data[key] = data[key].unsqueeze(0).to(self.device)  # 加入T维度
        return data




