
from collections import deque
from typing import Iterator, Union

from jax import random


class PRNGKeySequence(Iterator[random.PRNGKey]):
    '''Iterator of JAX random keys.
    from: https://github.com/deepmind/dm-haiku/blob/389fbe4ab164bbe5d2211218807b7f825cd59bdf/haiku/_src/base.py#L651-L726
    modified by: STomoya (https://github.com/STomoya/)
    '''
    def __init__(self, key_or_seed: Union[random.PRNGKey, int]) -> None:
        if isinstance(key_or_seed, int):
            self._key = random.PRNGKey(key_or_seed)
        else:
            self._key = key_or_seed
        self._subkeys = deque()

    def reserve(self, num):
        if num > 0:
            keys = random.split(self._key, num+1)
            self._key = keys[0]
            self._subkeys.extend(keys[1:])

    def __next__(self):
        if not self._subkeys:
            self.reserve(1)
        return self._subkeys.popleft()

    def next(self): return self.__next__()

    def take(self, num):
        self.reserve(max(num - len(self._subkeys), 0))
        return tuple(next(self) for _ in range(num))


if __name__=='__main__':
    keyseq = PRNGKeySequence(0)
    print(next(keyseq))
    print(keyseq.next())
    print(keyseq.take(2)[0])
