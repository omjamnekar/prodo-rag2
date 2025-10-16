import os
import time
import requests

# Integration tests that run only when RUN_INTEGRATION env var is set to '1'
BASE_URL = os.environ.get('BASE_URL') or 'http://localhost:8080'
RUN_INTEGRATION = os.environ.get('RUN_INTEGRATION') == '1'


def test_index_query_and_delete():
    if not RUN_INTEGRATION:
        print('Skipping integration test: set RUN_INTEGRATION=1 and BASE_URL to run')
        return

    repo_id = f'integration-test-{int(time.time())}'

    # 1. index
    index_payload = {
        'repoId': repo_id,
        'files': [{'filename': 'test.py', 'content': "print('Hello Integration')"}],
        'metadata': {}
    }
    r = requests.post(f'{BASE_URL}/rag/index', json=index_payload, timeout=60)
    assert r.status_code == 200, r.text
    print('Index response:', r.json())

    # 2. query
    query_payload = {'repoId': repo_id, 'prompt': 'What does this code do?', 'top_k': 5, 'metadata': {}}
    r = requests.post(f'{BASE_URL}/rag/query', json=query_payload, timeout=60)
    assert r.status_code == 200, r.text
    print('Query response:', r.json())

    # 3. delete
    r = requests.delete(f'{BASE_URL}/rag/delete', params={'repoId': repo_id}, timeout=60)
    assert r.status_code == 200, r.text
    print('Delete response:', r.json())
