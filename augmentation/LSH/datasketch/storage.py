"""
Storage backends for LSH index structures.

Provides dict-based storage implementations for in-memory LSH indices.
Can be extended to support Redis or other backends if needed.
"""

import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Set


def _random_name(length: int) -> bytes:
    """Generate a random name for storage identification."""
    return ''.join(random.choice(string.ascii_lowercase)
                   for _ in range(length)).encode('utf8')


def ordered_storage(config: Dict[str, Any], name: Optional[bytes] = None):
    """
    Return ordered storage system based on the specified config.

    The canonical example of such a storage container is
    ``defaultdict(list)``. Thus, the return value of this method contains
    keys and values. The values are ordered lists with the last added
    item at the end.

    Args:
        config (dict): Defines the configurations for the storage.
            For in-memory storage, the config ``{'type': 'dict'}`` will
            suffice.
        name (bytes, optional): A reference name for this storage container.
            For dict-type containers, this is ignored.

    Returns:
        OrderedStorage: An ordered storage instance
    """
    tp = config['type']
    if tp == 'dict':
        return DictListStorage(config)
    raise ValueError(f"Unknown storage type: {tp}")


def unordered_storage(config: Dict[str, Any], name: Optional[bytes] = None):
    """
    Return an unordered storage system based on the specified config.

    The canonical example of such a storage container is
    ``defaultdict(set)``. Thus, the return value of this method contains
    keys and values. The values are unordered sets.

    Args:
        config (dict): Defines the configurations for the storage.
            For in-memory storage, the config ``{'type': 'dict'}`` will
            suffice.
        name (bytes, optional): A reference name for this storage container.
            For dict-type containers, this is ignored.

    Returns:
        UnorderedStorage: An unordered storage instance
    """
    tp = config['type']
    if tp == 'dict':
        return DictSetStorage(config)
    raise ValueError(f"Unknown storage type: {tp}")


class Storage(ABC):
    """Base class for key, value containers where the values are sequences."""

    def __getitem__(self, key):
        return self.get(key)

    def __delitem__(self, key):
        return self.remove(key)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __contains__(self, item):
        return self.has_key(item)

    @abstractmethod
    def keys(self) -> Iterator:
        """Return an iterator on keys in storage."""
        return []

    @abstractmethod
    def get(self, key) -> Any:
        """Get list of values associated with a key.

        Returns empty list ([]) if `key` is not found.
        """
        pass

    def getmany(self, *keys) -> List:
        """Get values for multiple keys."""
        return [self.get(key) for key in keys]

    @abstractmethod
    def insert(self, key, *vals, **kwargs) -> None:
        """Add `val` to storage against `key`."""
        pass

    @abstractmethod
    def remove(self, *keys) -> None:
        """Remove `keys` from storage."""
        pass

    @abstractmethod
    def remove_val(self, key, val) -> None:
        """Remove `val` from list of values under `key`."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return size of storage with respect to number of keys."""
        pass

    @abstractmethod
    def itemcounts(self, **kwargs) -> Dict:
        """Returns the number of items stored under each key."""
        pass

    @abstractmethod
    def has_key(self, key) -> bool:
        """Determines whether the key is in the storage or not."""
        pass

    def status(self) -> Dict:
        """Return status information."""
        return {'keyspace_size': len(self)}

    def empty_buffer(self) -> None:
        """Empty any internal buffer (no-op for dict storage)."""
        pass

    def add_to_select_buffer(self, keys) -> None:
        """Query keys and add them to internal buffer."""
        if not hasattr(self, '_select_buffer'):
            self._select_buffer = self.getmany(*keys)
        else:
            self._select_buffer.extend(self.getmany(*keys))

    def collect_select_buffer(self) -> List:
        """Return buffered query results."""
        if not hasattr(self, '_select_buffer'):
            return []
        buffer = list(self._select_buffer)
        del self._select_buffer[:]
        return buffer


class OrderedStorage(Storage):
    """Storage where values maintain insertion order."""
    pass


class UnorderedStorage(Storage):
    """Storage where values are unordered (set-based)."""
    pass


class DictListStorage(OrderedStorage):
    """
    Wrapper class around ``defaultdict(list)`` enabling
    it to support an API consistent with `Storage`.
    """

    def __init__(self, config: Dict):
        self._dict: Dict[Any, List] = defaultdict(list)

    def keys(self) -> Iterator:
        return self._dict.keys()

    def get(self, key) -> List:
        return self._dict.get(key, [])

    def remove(self, *keys) -> None:
        for key in keys:
            if key in self._dict:
                del self._dict[key]

    def remove_val(self, key, val) -> None:
        if key in self._dict:
            self._dict[key].remove(val)

    def insert(self, key, *vals, **kwargs) -> None:
        self._dict[key].extend(vals)

    def size(self) -> int:
        return len(self._dict)

    def itemcounts(self, **kwargs) -> Dict:
        """Returns a dict where the keys are the keys of the container.
        The values are the *lengths* of the value sequences stored
        in this container.
        """
        return {k: len(v) for k, v in self._dict.items()}

    def has_key(self, key) -> bool:
        return key in self._dict


class DictSetStorage(UnorderedStorage, DictListStorage):
    """
    Wrapper class around ``defaultdict(set)`` enabling
    it to support an API consistent with `Storage`.
    """

    def __init__(self, config: Dict):
        self._dict: Dict[Any, Set] = defaultdict(set)

    def get(self, key) -> Set:
        return self._dict.get(key, set())

    def insert(self, key, vals, **kwargs) -> None:
        if isinstance(vals, (set, frozenset)):
            self._dict[key].update(vals)
        elif hasattr(vals, '__iter__') and not isinstance(vals, (str, bytes)):
            self._dict[key].update(vals)
        else:
            self._dict[key].add(vals)
