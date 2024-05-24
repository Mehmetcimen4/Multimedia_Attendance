from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory('.', 'Multimedia_UI_Design.html')

@app.route("/register_student", methods=["POST"])
def register_student():
    try:
        student_name = request.form.get("student_name")
        student_id = request.form.get("student_id")

        # Run TakePhotos.py with the provided student name and ID
        subprocess.run(["python", "TakePhotos.py", student_name, student_id])

        return jsonify({"status": "success", "message": "Student registered successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/upload_video", methods=["POST"])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})

        video_file = request.files['video']
        student_name = request.form.get("student_name")
        student_id = request.form.get("student_id")
        if video_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        # Save the uploaded video to a temporary file
        upload_folder = app.config['UPLOAD_FOLDER']
        video_path = os.path.join(upload_folder, f"video_{student_name}_{datetime.now().strftime('%m_%d_%H%M')}.mp4")
        video_file.save(video_path)

        # Run the video processing script
        subprocess.run(["python", "Video_for_Registeration.py", video_path, student_name, student_id])

        return jsonify({"status": "success", "message": "Video uploaded and processed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/webcam_for_attendance', methods=['POST'])
def webcam_for_attendance():
    try:
        result = subprocess.run(['python', 'Recognize.py'], capture_output=True, text=True)

        if result.returncode == 0:
            return jsonify({"message": "Attendance script executed successfully."})
        else:
            return jsonify({"message": f"Error executing attendance script: {result.stderr}"}), 500
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

@app.route("/upload_video_for_attendance", methods=["POST"])
def upload_video_for_attendance():
    try:
        if 'video_att' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})

        video_file = request.files['video_att']
        
        if video_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        # Save the uploaded video to a temporary file
        upload_folder = app.config['UPLOAD_FOLDER']
        video_path = os.path.join(upload_folder, f"video_{datetime.now().strftime('%m_%d_%H%M')}.mp4")
        video_file.save(video_path)

        # Run the video processing script
        subprocess.run(["python", "Recognize.py", video_path])

        return jsonify({"status": "success", "message": "Video uploaded and processed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/run_show', methods=['POST'])
def run_show():
    try:
        # Check if CSV file is in the request
        if 'csvFile' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})

        csv_file = request.files['csvFile']

        if csv_file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        # Save the uploaded CSV to a temporary file
        upload_folder = app.config['UPLOAD_FOLDER']
        csv_path = os.path.join(upload_folder, csv_file.filename)
        csv_file.save(csv_path)

        # Run the Show.py script with the CSV file path
        result = subprocess.run(['python', 'Show.py', csv_path], capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({'message': 'Script executed successfully.', 'output': result.stdout})
        else:
            return jsonify({'message': 'Script execution failed.', 'error': result.stderr}), 500
    except Exception as e:
        return jsonify({'message': 'An error occurred.', 'error': str(e)}), 500
    

if __name__ == "__main__":
    app.config['UPLOAD_FOLDER'] = 'uploads'
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
