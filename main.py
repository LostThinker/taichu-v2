from algorithms.base_algo import BaseAlgo
from collector.base_collector import BaseCollector
from commander.base_commander import GeneralBaseCommander
from utils.utils import make_policy


def main():
    env_args = None
    policy_args = None

    group_num = env_args.get('group_num')
    group_policy_args = []
    for i in range(group_num):
        group_policy_args.append(policy_args)
    commander = GeneralBaseCommander(env_args, group_policy_args)
    collector = BaseCollector(env_args)
    data = collector.collect()

