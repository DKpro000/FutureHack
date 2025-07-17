from flask import Flask, render_template, request, redirect, url_for, flash
import os
import json
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "static/uploads"
ANALYSIS_OUTPUT_PATH = r"routes\\to\\Your\\out.json\\path"
DOG_EMOTION_SCRIPT = r"routes\\to\\Your\\DogEmotion.py\\path"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        flash("No video file provided", "error")
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        flash("Empty filename", "error")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        subprocess.run(
            ["python", DOG_EMOTION_SCRIPT, save_path, "--output", ANALYSIS_OUTPUT_PATH],
            check=True
        )
    except subprocess.CalledProcessError as e:
        flash("Failed to analyze video", "error")
        return redirect(url_for('index'))

    return redirect(url_for('show_results'))

@app.route("/results")
def show_results():
    try:
        with open(ANALYSIS_OUTPUT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = {
            "success": True,
            "num_frames": len(data.get("detailed_emotions", [])),
            "scenes": data.get("scene_context", []),
            "emotions": data.get("detailed_emotions", []),
            "audio": {
                "classification": data.get("audio_behavior", "Unknown"),
                "confidence": data.get("audio_behavior_confidence", 0.0),
                "all_scores": data.get("audio_details", {}).get("all_probabilities_class", {})
            },
            "dog_thoughts": data.get("dog_thoughts")
        }
    except Exception as e:
        results = {
            "success": False,
            "error": str(e)
        }

    return render_template("results.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
