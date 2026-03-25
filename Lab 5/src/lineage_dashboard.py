from flask import Flask, render_template, jsonify
from src.mlmd_tracker import PipelineTracker
from src.utils import get_logger
import os

logger = get_logger(__name__)
app = Flask(__name__)

def get_tracker():
    return PipelineTracker()

@app.route("/")
def index():
    return render_template("lineage.html")

@app.route("/api/runs")
def get_runs():
    try:
        tracker = get_tracker()
        return jsonify(tracker.get_all_pipeline_runs())
    except Exception as e:
        logger.error(f"Error fetching runs: {e}")
        return jsonify([])

@app.route("/api/lineage")
def get_full_lineage():
    try:
        tracker = get_tracker()
        return jsonify(tracker.get_full_lineage_graph())
    except Exception as e:
        logger.error(f"Error fetching lineage: {e}")
        return jsonify({"nodes": [], "edges": []})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=False)
