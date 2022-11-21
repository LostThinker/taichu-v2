import numpy as np
from typing import Union, Any, Optional, List


class BaseReplayBuffer:
    def __init__(
            self,
            size,
    ):
        super(BaseReplayBuffer, self).__init__()
        self._size = size
        self._push_count = 0
        self._buffer = [None for _ in range(size)]  # [{(T, A N)},...,{(T, A, N)}]
        self._tail = 0

        # ["obs", "action", "reward", "last_action", "prev_state", "agent_id", "global_state", "action_mask"]
    def push(self, data):
        if isinstance(data, list):
            self._extend(data)
        elif 'AutoProxy' in str(type(data)):
            self._extend(data)
        else:
            self._extend(data)

    def _extend(self, data):
        length = len(data)
        self._push_count += length
        if self._tail + length <= self._size:
            self._buffer[self._tail:self._tail + length] = data
            self._tail = self._tail + length
        else:
            new_tail = length - (self._size - self._tail)
            self._buffer[self._tail:] = data[:self._size - self._tail]
            self._buffer[:new_tail] = data[self._size - self._tail:]
            self._tail = new_tail

    def _append(self, data):
        self._push_count += 1
        self._buffer[self._tail] = data
        self._tail = (self._tail + 1) % self._size

    def sample(self, size=64, replace=False):
        if self._push_count < size:
            return None
        indices = self._get_indices(size, replace=replace)
        sample_data = self._sample_with_indices(indices)
        return sample_data

    def _get_indices(self, size, replace=False):
        if self._push_count >= self._size:
            tail = self._size
        else:
            tail = self._tail
        indices = list(np.random.choice(a=tail, size=size, replace=replace))
        return indices

    def _sample_with_indices(self, indices):
        data = []
        for idx in indices:
            assert self._buffer[idx] is not None
            data.append(self._buffer[idx])
        return data

