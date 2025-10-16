import os
from pymongo import MongoClient

MONGODB_URI = os.getenv('MONGODB_URI')
_client = MongoClient(MONGODB_URI)
_db = _client.get_database('ragsvc')

def save_index_metadata(repo_id: str, data: dict):
    _db.indexes.update_one({'repoId': repo_id}, {'$set': {'repoId': repo_id, 'data': data}}, upsert=True)

def save_query_log(repo_id: str, log: dict):
    _db.query_logs.insert_one({'repoId': repo_id, 'log': log})