from flask import Flask
import numpy as np
from dotenv import load_dotenv
from flask import request, jsonify
from piplines.rag_pipeline import process_rag, index_repo, reset_repo
import traceback
import asyncio


load_dotenv()
app = Flask(__name__)

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
        data = request.get_json(force=True)
        req = parse_query_request(data)
        # Call async function from sync Flask
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_rag(req["repoId"], req["prompt"], req["top_k"], req["metadata"]))
            safe_result = convert_ndarray_to_list(result)
            return jsonify(safe_result)
        finally:
            loop.close()
    except Exception as e:
        print("Error in /rag/query:", e, flush=True)
        print(traceback.format_exc(), flush=True)
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/rag/reset', methods=['DELETE'])
def rag_reset():
    try:
        repoId = request.args.get('repoId')
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(reset_repo(repoId)) # type: ignore
            return jsonify({"status": "reset", "repoId": repoId})
        finally:
            loop.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
            
            loop.run_until_complete(reset_repo(repoId))
            return jsonify({"status": "deleted", "repoId": repoId})
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
        data = request.get_json(force=True)
        print(f"Received data for /rag/index: {data}", flush=True)
        req = parse_index_request(data)
        repo_id = req["repoId"]
        files = req["files"]
        metadata = req["metadata"]

        if not repo_id or not files or not isinstance(files, list):
            raise ValueError("Missing or invalid repoId/files in request.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(index_repo(repo_id, files, metadata)) 
            print(f"Indexing result for repoId {repo_id}: {result}", flush=True)
            return jsonify({"success": True, "result": result})
        finally:
            loop.close()
    except Exception as e:
        tb = traceback.format_exc()
        print(f"Error in /rag/index: {e}\n{tb}", flush=True)
        return jsonify({"error": str(e), "traceback": tb}), 500



@app.route('/')
def home():
    return "Hello, PRODO RAG on Vercel!"

if __name__ == '__main__':
    app.run(debug=True)