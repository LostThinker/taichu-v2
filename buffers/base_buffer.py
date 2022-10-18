import numpy as np
from abc import ABC, abstractmethod


class BaseBuffer(ABC):
    def __init__(self, buffer_size):
        self.buffer = []

    def put(self, data):
        self.buffer.append(data)

    def put_batch(self, batch_data):
        self.buffer.extend(batch_data)

    def get(self, batch_size):
        """
        从buffer中随机采样，需实现
        """
        data = []
        idx = np.random.choice(batch_size, size=batch_size, replace=False)
        for i in idx:
            data.append(self.buffer[i])
        data = self._batch_data_process(data)
        return data

    def _batch_data_process(self, data):
        # 将time_step维度整合到dict内，使数据变为（T,E,A,N），T为time_step，E为env_num，A为agent_num
        pass
