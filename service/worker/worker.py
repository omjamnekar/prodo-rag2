import threading
import queue
import time
import uuid
import logging
import inspect
from typing import Optional

from service.piplines.rag_pipeline import index_repo
from service.db import database

logger = logging.getLogger(__name__)


class IndexWorker:
    def __init__(self, num_workers=1):
        self.q = queue.Queue()
        self.status = {}  # job_id -> {'status': str, 'meta': ...}
        self.threads = []
        self.running = False
        self.num_workers = num_workers

    def start(self):
        if self.running:
            return
        self.running = True
        for _ in range(self.num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        self.running = False
        # put None sentinel for each thread
        for _ in range(len(self.threads)):
            self.q.put(None)
        # join threads to ensure clean shutdown
        for t in self.threads:
            try:
                if t.is_alive():
                    t.join(timeout=5)
            except Exception:
                pass

    def submit(self, repo_id: str, files, metadata=None) -> str:
        job_id = str(uuid.uuid4())
        self.status[job_id] = {'status': 'queued', 'repo_id': repo_id, 'result': None, 'error': None, 'created_at': time.time()}
        self.q.put((job_id, repo_id, files, metadata))
        # Optionally persist job to DB
        try:
            database.save_index_job(job_id, repo_id, metadata or {})
        except Exception:
            pass
        return job_id

    def get_status(self, job_id: str):
        return self.status.get(job_id)

    def _worker_loop(self):
        while self.running:
            item = self.q.get()
            if item is None:
                break
            job_id, repo_id, files, metadata = item
            try:
                # call index_repo (async) from sync thread
                import asyncio
                from functools import partial
                loop = asyncio.new_event_loop()
                try:
                    # support both async and sync implementations of index_repo:
                    if inspect.iscoroutinefunction(index_repo):
                        coro = index_repo(repo_id, files, metadata or {})
                        res = loop.run_until_complete(coro)
                    else:
                        # run synchronous index_repo in a thread executor so run_until_complete
                        # always receives an awaitable
                        func = partial(index_repo, repo_id, files, metadata or {})
                        res = loop.run_until_complete(loop.run_in_executor(None, func))
                finally:
                    try:
                        loop.stop()
                    except Exception:
                        pass
                    try:
                        loop.close()
                    except Exception:
                        pass
                        pass
                self.status[job_id]['status'] = 'completed'
                self.status[job_id]['result'] = res
                try:
                    if inspect.iscoroutine(res):
                        res = loop.run_until_complete(res)
                    database.update_index_job_result(job_id, res)
                except Exception:
                    pass
            except Exception as e:
                logger.exception('Index job failed')
                self.status[job_id]['status'] = 'failed'
                self.status[job_id]['error'] = str(e)
                try:
                    database.update_index_job_error(job_id, str(e))
                except Exception:
                    pass
        # worker exiting, attempt to persist or cleanup if needed
        logger.info('IndexWorker threads exiting')
