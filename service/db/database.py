import os
from pymongo import MongoClient

MONGODB_URI = os.getenv('MONGODB_URI')
_client = None
_db = None
if MONGODB_URI:
    try:
        _client = MongoClient(MONGODB_URI)
        _db = _client.get_database('ragsvc')
    except Exception:
        _client = None
        _db = None

def save_index_metadata(repo_id: str, data: dict):
    if _db is None:
        raise RuntimeError("MongoDB is not configured or failed to initialize.")
    _db.indexes.update_one({'repoId': repo_id}, {'$set': {'repoId': repo_id, 'data': data}}, upsert=True)

def save_query_log(repo_id: str, log: dict):
    if _db is None:
        raise RuntimeError("MongoDB is not configured or failed to initialize.")
    _db.query_logs.insert_one({'repoId': repo_id, 'log': log})


def save_index_job(job_id: str, repo_id: str, meta: dict):
    if _db is None:
        return  # Skip operation if DB is not initialized
    """Persist an index job record. Non-fatal on errors."""
    try:
        _db.index_jobs.update_one({'job_id': job_id}, {'$set': {'job_id': job_id, 'repo_id': repo_id, 'meta': meta, 'status': 'queued'}}, upsert=True)
    except Exception:
        pass


def update_index_job_result(job_id: str, result: dict):
    if _db is None:
        return  # Skip operation if DB is not initialized
    try:
        _db.index_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'completed', 'result': result}})
    except Exception:
        pass


def update_index_job_error(job_id: str, error: str):
    if _db is None:
        return  # Skip operation if DB is not initialized
    try:
        _db.index_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'failed', 'error': error}})
    except Exception:
        pass


def shutdown():
    """Close MongoDB client if open to release sockets and resources."""
    global _client
    try:
        if _client is not None:
            _client.close()
    except Exception:
        pass
    _client = None