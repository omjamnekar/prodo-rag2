import time
import threading

class TTLCache:
    def __init__(self, ttl_seconds=300, max_items=1024):
        self.ttl = ttl_seconds
        self.max_items = max_items
        self.lock = threading.Lock()
        self.store = {}  # key -> (value, expires_at)

    def _evict_if_needed(self):
        if len(self.store) <= self.max_items:
            return
        # evict oldest
        items = sorted(self.store.items(), key=lambda kv: kv[1][1])
        while len(items) > self.max_items:
            k, _ = items.pop(0)
            self.store.pop(k, None)

    def get(self, key):
        with self.lock:
            entry = self.store.get(key)
            if not entry:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self.store[key]
                return None
            return value

    def set(self, key, value):
        with self.lock:
            self.store[key] = (value, time.time() + self.ttl)
            self._evict_if_needed()
