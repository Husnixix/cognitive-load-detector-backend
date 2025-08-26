from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from dotenv import load_dotenv, find_dotenv

from app.controller.schemas import cognitive_state_to_dict
from app.application.services.cognitive_load_service import CognitiveLoadService
from app.domain.services.cognitive_load_analyzer import CognitiveLoadAnalyzer
from app.infrastructure.repositories.mongo_cognitive_state_repository import MongoCognitiveStateRepository
from app.infrastructure.capture.opencv_facial_cue_detector import OpenCVFacialCueDetector
from app.infrastructure.capture.keystroke_tracker_impl import SystemKeystrokeTracker
from app.infrastructure.time.system_clock import SystemClock

app = Flask(__name__)
CORS(app)

# Load env variables from nearest .env regardless of CWD
load_dotenv(find_dotenv())

# Single service instance for API reads. We do NOT start detection here to avoid side effects.
service = CognitiveLoadService(
    repository=MongoCognitiveStateRepository(),
    facial_detector=OpenCVFacialCueDetector(),
    keystroke_tracker=SystemKeystrokeTracker(),
    analyzer=CognitiveLoadAnalyzer(),
    clock=SystemClock(),
)


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


@app.post("/start")
def start_detection():
    try:
        # Avoid double-start
        if getattr(service, "is_detecting", False):
            return jsonify({"success": True, "message": "Detection already running"}), 200

        service.start_detecting_cognitive_load()
        return jsonify({"success": True, "message": "Detection started"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.post("/stop")
def stop_detection():
    try:
        if not getattr(service, "is_detecting", False):
            return jsonify({"success": True, "message": "Detection already stopped"}), 200

        service.stop_detecting_cognitive_load()
        return jsonify({"success": True, "message": "Detection stopped"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Run as a standalone API server
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.run(host="0.0.0.0", port=5000, debug=False)
