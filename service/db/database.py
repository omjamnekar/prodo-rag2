import os
from pymongo import MongoClient

MONGODB_URI = os.getenv('MONGODB_URI')
_client = MongoClient(MONGODB_URI)
_db = _client.get_database('ragsvc')

def save_index_metadata(repo_id: str, data: dict):
    _db.indexes.update_one({'repoId': repo_id}, {'$set': {'repoId': repo_id, 'data': data}}, upsert=True)

def save_query_log(repo_id: str, log: dict):
    _db.query_logs.insert_one({'repoId': repo_id, 'log': log})


def save_index_job(job_id: str, repo_id: str, meta: dict):
    """Persist an index job record. Non-fatal on errors."""
    try:
        _db.index_jobs.update_one({'job_id': job_id}, {'$set': {'job_id': job_id, 'repo_id': repo_id, 'meta': meta, 'status': 'queued'}}, upsert=True)
    except Exception:
        pass


def update_index_job_result(job_id: str, result: dict):
    try:
        _db.index_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'completed', 'result': result}})
    except Exception:
        pass


def update_index_job_error(job_id: str, error: str):
    try:
        _db.index_jobs.update_one({'job_id': job_id}, {'$set': {'status': 'failed', 'error': error}})
    except Exception:
        pass