# routes.py
import os
import json
import uuid
import subprocess
from flask import request, redirect, flash, url_for, render_template, send_file
from werkzeug.utils import secure_filename
from app import app

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
WORKING_DIR = r"\routes\to\Project_Folder"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        "success": True,
        "num_frames": len(raw.get("detailed_emotions", [])),
        "audio_length": 0.0,
        "scenes": raw.get("scene_context", []),
        "emotions": [
            {
                "emotion": e["emotion"],
                "confidence": e["confidence"],
                "all_probabilities": e["all_probabilities"]
            }
            for e in raw.get("detailed_emotions", [])
        ],
        "audio": {
            "classification": raw.get("audio_details", {}).get("predicted_class", "No audio"),
            "confidence": raw.get("audio_details", {}).get("class_confidence", 0.0),
            "all_scores": raw.get("audio_details", {}).get("all_probabilities_class", {}),
            "all_phrases": raw.get("audio_details", {}).get("all_probabilities_phrase", {})
        },
        "dog_thoughts": raw.get("dog_thoughts"),
        "spectrogram_path": None
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash('Invalid file type.', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    save_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    file.save(save_path)

    result_id = str(uuid.uuid4())
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{result_id}_out.json")

    try:
        subprocess.run([
            "python", "DogEmotion.py", save_path, "--output", output_path
        ], cwd=WORKING_DIR, check=True)

        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Analysis output not found: {output_path}")

        results = parse_results(output_path)
        return render_template("results.html", results=results, results_id=result_id)

    except subprocess.CalledProcessError as e:
        flash(f"Error running DogEmotion.py: {e}", "error")
    except Exception as e:
        return render_template("results.html", results={"success": False, "error": str(e)})

    return redirect(url_for('index'))

@app.route('/download/<results_id>')
def download_results(results_id):
    path = os.path.join(app.config['UPLOAD_FOLDER'], f"{results_id}_out.json")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="analysis_results.json")
    flash("Result file not found", "error")
    return redirect(url_for('index'))
