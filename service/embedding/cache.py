import os
import hashlib
import threading
import pickle
from collections import OrderedDict
from time import time

class LRUCache:
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self.lock = threading.Lock()
        self.cache = OrderedDict()

    def _evict_if_needed(self):
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def get(self, key):
        with self.lock:
            v = self.cache.get(key)
            if v is None:
                return None
            # move to end
            self.cache.move_to_end(key)
            return v

    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            self._evict_if_needed()


class DiskCache:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # file-based map of key->pickle

    def _path_for_key(self, key):
        return os.path.join(self.path, f"{key}.pkl")

    def get(self, key):
        p = self._path_for_key(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key, value):
        p = self._path_for_key(key)
        try:
            with open(p, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            pass


# Simple cache manager combining LRU in-memory with disk fallback
class EmbeddingCache:
    def __init__(self, max_memory_items=4096, disk_path=None):
        self.mem = LRUCache(max_size=max_memory_items)
        self.disk = DiskCache(disk_path) if disk_path else None

    @staticmethod
    def _key_for_text(text: str):
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return h

    def get(self, text: str):
        key = self._key_for_text(text)
        v = self.mem.get(key)
        if v is not None:
            return v
        if self.disk:
            v = self.disk.get(key)
            if v is not None:
                # warm memory cache
                self.mem.set(key, v)
            return v
        return None

    def set(self, text: str, embedding):
        key = self._key_for_text(text)
        self.mem.set(key, embedding)
        if self.disk:
            self.disk.set(key, embedding)
