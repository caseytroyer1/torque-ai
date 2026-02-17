from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Allowed video file extensions
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm'}

# Maximum file size (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_angle(point1, point2, point3):
    """Calculate angle between three points (point2 is the vertex)"""
    try:
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    except:
        return None


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    try:
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    except:
        return None


def analyze_video_with_mediapipe(video_path):
    """
    Analyze golf swing video using MediaPipe Pose detection.
    
    Returns tuple: (analysis_data, error_message)
    """
    try:
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Could not open video file"
        
        frame_count = 0
        landmarks_data = []
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks_data.append({
                    'frame': frame_count,
                    'landmarks': results.pose_landmarks
                })
            
            frame_count += 1
        
        cap.release()
        pose.close()
        
        if not landmarks_data:
            return None, "No pose detected in video"
        
        # Return simple analysis for now
        return {
            'frames_analyzed': frame_count,
            'poses_detected': len(landmarks_data),
            'analysis': 'Basic pose tracking successful'
        }, None
        
    except Exception as e:
        return None, str(e)


@app.route('/analyze-swing', methods=['POST'])
def analyze_swing():
    """Handle video file upload for swing analysis"""
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please upload a video file'
            }), 400
        
        file = request.files['file']
        
        # Check if file was actually selected
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'message': 'Please select a video file to upload'
            }), 400
        
        # Validate file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'message': f'File must be one of: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Get file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)  # Reset file pointer
        
        # Check file size
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'error': 'File too large',
                'message': f'File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024):.0f}MB'
            }), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Analyze video with MediaPipe
        analysis_data, analysis_error = analyze_video_with_mediapipe(file_path)
        
        # Prepare response
        response_data = {
            'success': True,
            'message': 'Video uploaded and analyzed successfully',
            'filename': filename,
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2)
        }
        
        # Add analysis results if available
        if analysis_error:
            response_data['analysis_error'] = analysis_error
            response_data['analysis'] = None
        else:
            response_data['analysis'] = analysis_data
            response_data['analysis_error'] = None
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)

