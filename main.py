from flask import Flask
import os
import numpy as np
from dotenv import load_dotenv
from flask import request, jsonify
from service.piplines.rag_pipeline import process_rag, index_repo, reset_repo
from service.worker.worker import IndexWorker
from service.cache.query_cache import TTLCache
import hashlib
import traceback
import asyncio
from threading import Semaphore


load_dotenv()
app = Flask(__name__)

# Limit concurrent heavy requests to avoid memory spikes. Default 2 concurrent.
MAX_CONCURRENCY = int(os.environ.get('MAX_CONCURRENCY', '2'))
_semaphore = Semaphore(MAX_CONCURRENCY)

# optional background worker
_background_index = os.environ.get('BACKGROUND_INDEX', 'false').lower() in ('1', 'true', 'yes')
_worker: IndexWorker | None = None
if _background_index:
    _worker = IndexWorker(num_workers=int(os.environ.get('INDEX_WORKERS', '1')))
    _worker.start()

# query cache
_query_cache = TTLCache(ttl_seconds=int(os.environ.get('QUERY_CACHE_TTL', '300')), max_items=int(os.environ.get('QUERY_CACHE_ITEMS', '1024')))

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(v) for v in obj]
    else:
        return obj


# Data validation helpers for Flask
def parse_repo_file(data):
    return {
        "filename": data.get("filename"),
        "content": data.get("content")
    }

def parse_index_request(data):
    return {
        "repoId": data.get("repoId"),
        "files": [parse_repo_file(f) for f in data.get("files", [])],
        "metadata": data.get("metadata", {})
    }

def parse_query_request(data):
    return {
        "repoId": data.get("repoId"),
        "prompt": data.get("prompt"),
        "top_k": data.get("top_k", 6),
        "metadata": data.get("metadata", {})
    }


@app.route('/rag/query', methods=['POST'])
def rag_query():
    try:
        # try to acquire semaphore quickly (avoid waiting indefinitely)
        acquired = _semaphore.acquire(timeout=0.5)
        if not acquired:
            return jsonify({"error": "Too many concurrent requests"}), 429
        data = request.get_json(force=True)
        req = parse_query_request(data)
        # Call async function from sync Flask
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # check query cache
            cache_key = hashlib.sha256((req["repoId"] + '||' + req["prompt"] + '||' + str(req["top_k"])).encode('utf-8')).hexdigest()
            cached = _query_cache.get(cache_key)
            if cached:
                result = cached
            else:
                result = loop.run_until_complete(process_rag(req["repoId"], req["prompt"], req["top_k"], req["metadata"]))
                try:
                    _query_cache.set(cache_key, result)
                except Exception:
                    pass
            safe_result = convert_ndarray_to_list(result)
            return jsonify(safe_result)
        finally:
            try:
                _semaphore.release()
            except Exception:
                pass
            loop.close()
    except Exception as e:
        print("Error in /rag/query:", e, flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/rag/reset', methods=['POST'])
def rag_reset():
    try:
        acquired = _semaphore.acquire(timeout=0.5)
        if not acquired:
            return jsonify({"error": "Too many concurrent requests"}), 429
        data = request.get_json(force=True)
        req = parse_index_request(data)
        repo_id = req.get('repoId')
        files = req.get('files')
        metadata = req.get('metadata', {})

        if not repo_id:
            return jsonify({"error": "repoId required"}), 400

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(reset_repo(repo_id, files, metadata)) # type: ignore
            return jsonify({"status": "reset", "repoId": repo_id, "result": result})
        finally:
            try:
                _semaphore.release()
            except Exception:
                pass
            loop.close()
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500


# Delete all vectors for a repo (namespace)
@app.route('/rag/delete', methods=['DELETE'])
def delete_repo_vectors():
    try:
        repoId = request.args.get('repoId')
        if not repoId:
            return jsonify({"error": "repoId required"}), 400
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # call delete_repo which performs namespace deletion
            from service.piplines.rag_pipeline import delete_repo
            result = loop.run_until_complete(delete_repo(repoId))
            return jsonify({"status": "deleted", "repoId": repoId, "result": result})
        finally:
            loop.close()
    except Exception as e:
        print(f"Error in /rag/delete: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/rag/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})


@app.route('/rag/index', methods=['POST'])
def index_repo_call():
    try:
        acquired = _semaphore.acquire(timeout=0.5)
        if not acquired:
            return jsonify({"error": "Too many concurrent requests"}), 429
        data = request.get_json(force=True)
        print(f"Received data for /rag/index: {data}", flush=True)
        req = parse_index_request(data)
        repo_id = req["repoId"]
        files = req["files"]
        metadata = req["metadata"]

        if not repo_id or not files or not isinstance(files, list):
            raise ValueError("Missing or invalid repoId/files in request.")

        # If background indexing is enabled, enqueue and return job id
        if _worker:
            try:
                job_id = _worker.submit(repo_id, files, metadata)
                return jsonify({"success": True, "job_id": job_id, "background": True})
            finally:
                try:
                    _semaphore.release()
                except Exception:
                    pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(index_repo(repo_id, files, metadata)) 
            print(f"Indexing result for repoId {repo_id}: {result}", flush=True)
            return jsonify({"success": True, "result": result})
        finally:
            try:
                _semaphore.release()
            except Exception:
                pass
            loop.close()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in /rag/index: {e}\n{tb}", flush=True)
        return jsonify({"error": str(e), "traceback": tb}), 500



@app.route('/')
def home():
    return "Hello, PRODO RAG on Vercel!"

if __name__ == '__main__':
    # Bind to 0.0.0.0 and use the PORT environment variable so hosting platforms (Render, Vercel)
    # can detect the open port. Default to 8080 if not provided.
    port = int(os.environ.get('PORT', 8002))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
