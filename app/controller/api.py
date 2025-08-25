from flask import Flask, jsonify
from flask_cors import CORS
import logging

from app.application.services.cognitive_load_service import CognitiveLoadService
from app.controller.schemas import cognitive_state_to_dict

app = Flask(__name__)
CORS(app)

# Single service instance for API reads. We do NOT start detection here to avoid side effects.
service = CognitiveLoadService()


@app.get("/latest")
def get_latest():
    latest = service.get_latest()
    payload = cognitive_state_to_dict(latest)
    return jsonify({"data": payload, "success": True}), 200


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/history")
def get_history():
    history = service.get_history() or []
    payload = [cognitive_state_to_dict(item) for item in history]
    return jsonify({"data": payload, "count": len(payload), "success": True}), 200


if __name__ == "__main__":
    # Run as a standalone API server
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.run(host="0.0.0.0", port=5000, debug=False)
