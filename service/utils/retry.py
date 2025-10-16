import time
import functools

def retry(exception_types=(Exception,), tries=3, delay=0.5, backoff=2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _tries = tries
            _delay = delay
            last_exc = None
            while _tries > 0:
                try:
                    return func(*args, **kwargs)
                except exception_types as e:
                    last_exc = e
                    _tries -= 1
                    if _tries <= 0:
                        break
                    time.sleep(_delay)
                    _delay *= backoff
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"Retries exhausted for {func.__name__}")
        return wrapper
    return decorator
