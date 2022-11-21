def test():
    from algorithms.qmix import QMIXPolicy
    from algorithms.qmix import QMIX
    from algorithms.qmix import qmix_default_config
    from envs.vec_env_wrapper import VecEnvWrapper
    from env.smac_env import SC2Env
    env_args = dict(env_type='SC2', map_name="8m", difficulty="7", seed=0, max_step=500)
    default_configs = coma_default_config(env_args)
    model = COMA(**default_configs.model_config)
    coma_policy = COMAPolicy(model, default_configs.policy_config)
    vec_env = VecEnvWrapper(**default_configs.collect_config.vec_env_config)
    collector = BaseCollector(vec_env, coma_policy)
    data = collector.collect(sample_num=32, sample_type='unroll')
    train_info = coma_policy.learn(data)
    print(len(data))


if __name__ == '__main__':
    test()