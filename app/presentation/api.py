from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv
import logging
from app.application.cognitive_load_service import CognitiveLoadService
from app.infrastructure.entities.cognitive_load_entity import cognitive_state_to_dict

app = Flask(__name__)
CORS(app)

load_dotenv(find_dotenv())

service = CognitiveLoadService()
@app.post("/start")
def start_detection():
    try:
        # Avoid double-start
        if getattr(service, "is_detecting", False):
            return jsonify({"success": True, "message": "Detection already running"}), 200

        service.start_detection()
        return jsonify({"success": True, "message": "Detection started"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.post("/stop")
def stop_detection():
    try:
        if not getattr(service, "is_detecting", False):
            return jsonify({"success": True, "message": "Detection already stopped"}), 200

        service.stop_detection()
        return jsonify({"success": True, "message": "Detection stopped"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.get("/latest")
def get_latest():
    latest = service.get_latest_cognitive_state()
    payload = cognitive_state_to_dict(latest)
    return jsonify({"data": payload, "success": True}), 200

@app.get("/history")
def get_history():
    history = service.get_cognitive_state_history() or []
    payload = [cognitive_state_to_dict(item) for item in history]
    return jsonify({"data": payload, "count": len(payload), "success": True}), 200

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # Run as a standalone API server
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.run(host="0.0.0.0", port=5000, debug=False)