from algorithms.base_algo import BaseAlgo
from collector.base_collector import BaseCollector
from commander.base_commander import GeneralBaseCommander
from utils.utils import make_policy
from abc import ABC, abstractmethod


class BaseRunner(ABC):
    def __init__(self, env_args, policy_args, collector_args):
        self.collector = BaseCollector(collector_args)
        self.commander = GeneralBaseCommander(env_args, policy_args)

    def run(self):
        while True:
            data = self.collector.collect(self.commander, batch_size=10, collect_type='step')
            train_log = self.commander.train(data)


class BaseParallelRunner(ABC):
    def __init__(self, env_args, policy_args, collector_args):
        self.collector = BaseParallelCollector(collector_args)
        self.commander = GeneralParallelBaseCommander(env_args, policy_args)

    def run(self):
        self.collector.start()
        self.commander.start()

