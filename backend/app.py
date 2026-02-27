from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import numpy as np
from database import init_database, create_user, get_user_by_email, get_user_by_id, verify_password
from auth import create_session, verify_session, delete_session

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


def calculate_angle_3d(point1, point2, point3):
    """Calculate angle at point2 given three points"""
    import math
    # Vector from point2 to point1
    v1 = [point1.x - point2.x, point1.y - point2.y, point1.z - point2.z]
    # Vector from point2 to point3
    v2 = [point3.x - point2.x, point3.y - point2.y, point3.z - point2.z]
    
    # Dot product and magnitudes
    dot = sum(a*b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a*a for a in v1))
    mag2 = math.sqrt(sum(a*a for a in v2))
    
    # Angle in degrees
    if mag1 * mag2 == 0:
        return 0
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
    return math.degrees(math.acos(cos_angle))


def analyze_video_with_mediapipe(video_path):
    """Analyze golf swing video using MediaPipe Pose detection"""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import cv2
        import math
        
        # Set up MediaPipe Pose Landmarker
        base_options = python.BaseOptions(model_asset_path='backend/pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        detector = vision.PoseLandmarker.create_from_options(options)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        poses_detected = 0
        
        # Store key frames: address (first), backswing (middle), impact (60%)
        key_frames = {
            'address': None,
            'backswing': None,
            'impact': None
        }
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        address_frame = 0
        backswing_frame = total_frames // 2
        impact_frame = int(total_frames * 0.6)
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to MediaPipe format
            timestamp_ms = int((frame_count / fps) * 1000) if fps > 0 else frame_count * 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect pose
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                poses_detected += 1
                landmarks = detection_result.pose_landmarks[0]
                
                # Store key frames
                if frame_count == address_frame:
                    key_frames['address'] = landmarks
                elif frame_count == backswing_frame:
                    key_frames['backswing'] = landmarks
                elif frame_count == impact_frame:
                    key_frames['impact'] = landmarks
            
            frame_count += 1
        
        cap.release()
        detector.close()
        
        # Calculate angles for key frames
        # MediaPipe pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
        def calculate_hip_z_rotation(landmarks):
            """Calculate hip rotation using Z-axis depth difference (frontal plane rotation)"""
            if not landmarks or len(landmarks) <= max(LEFT_HIP, RIGHT_HIP):
                return None
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]
            # Z-axis depth difference: positive when right hip is forward (toward camera)
            hip_z_rotation = (right_hip.z - left_hip.z) * 100
            return hip_z_rotation
        
        def calculate_shoulder_z_rotation(landmarks):
            """Calculate shoulder rotation using Z-axis depth difference (frontal plane rotation)"""
            if not landmarks or len(landmarks) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
                return None
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            # Z-axis depth difference: positive when right shoulder is forward (toward camera)
            shoulder_z_rotation = (right_shoulder.z - left_shoulder.z) * 100
            return shoulder_z_rotation
        
        def calculate_spine_angle(landmarks):
            """Calculate spine angle (angle from hip midpoint to shoulder midpoint relative to vertical)"""
            if not landmarks or len(landmarks) <= max(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP):
                return None
            left_shoulder = landmarks[LEFT_SHOULDER]
            right_shoulder = landmarks[RIGHT_SHOULDER]
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]
            
            # Calculate midpoints
            shoulder_mid = {
                'x': (left_shoulder.x + right_shoulder.x) / 2,
                'y': (left_shoulder.y + right_shoulder.y) / 2,
                'z': (left_shoulder.z + right_shoulder.z) / 2
            }
            hip_mid = {
                'x': (left_hip.x + right_hip.x) / 2,
                'y': (left_hip.y + right_hip.y) / 2,
                'z': (left_hip.z + right_hip.z) / 2
            }
            
            # Calculate angle from vertical (90 degrees = straight up)
            dx = shoulder_mid['x'] - hip_mid['x']
            dy = shoulder_mid['y'] - hip_mid['y']
            # Vertical is 90 degrees, so we calculate deviation from vertical
            angle = math.degrees(math.atan2(dx, -dy))  # Negative dy because y increases downward
            return round(angle, 1)
        
        # Calculate Z-axis depth differences at ADDRESS (baseline)
        baseline_hip_z = calculate_hip_z_rotation(key_frames['address']) if key_frames['address'] else None
        baseline_shoulder_z = calculate_shoulder_z_rotation(key_frames['address']) if key_frames['address'] else None
        
        # Calculate Z-axis depth differences at BACKSWING
        backswing_hip_z = calculate_hip_z_rotation(key_frames['backswing']) if key_frames['backswing'] else None
        backswing_shoulder_z = calculate_shoulder_z_rotation(key_frames['backswing']) if key_frames['backswing'] else None
        
        # Calculate Z-axis depth differences at IMPACT
        impact_hip_z = calculate_hip_z_rotation(key_frames['impact']) if key_frames['impact'] else None
        impact_shoulder_z = calculate_shoulder_z_rotation(key_frames['impact']) if key_frames['impact'] else None
        
        # Calculate rotation amounts (delta from baseline using Z-axis depth)
        backswing_hip_rotation = round(abs(backswing_hip_z - baseline_hip_z), 1) if baseline_hip_z is not None and backswing_hip_z is not None else None
        print(f"DEBUG Hip Rotation Calculation (Z-axis):")
        print(f"  baseline_hip_z: {baseline_hip_z}")
        print(f"  backswing_hip_z: {backswing_hip_z}")
        print(f"  backswing_hip_rotation (delta): {backswing_hip_rotation}°")
        print(f"  Expected: 30-60°")
        print("=" * 50)
        
        backswing_shoulder_rotation = round(abs(backswing_shoulder_z - baseline_shoulder_z), 1) if baseline_shoulder_z is not None and backswing_shoulder_z is not None else None
        print(f"DEBUG Shoulder Rotation Calculation (Z-axis):")
        print(f"  baseline_shoulder_z: {baseline_shoulder_z}")
        print(f"  backswing_shoulder_z: {backswing_shoulder_z}")
        print(f"  backswing_shoulder_rotation (delta): {backswing_shoulder_rotation}°")
        print(f"  Expected: 80-110°")
        print("=" * 50)
        
        impact_hip_rotation = round(abs(impact_hip_z - baseline_hip_z), 1) if baseline_hip_z is not None and impact_hip_z is not None else None
        impact_shoulder_rotation = round(abs(impact_shoulder_z - baseline_shoulder_z), 1) if baseline_shoulder_z is not None and impact_shoulder_z is not None else None
        
        # Calculate spine angle (unchanged - this is still a valid measurement)
        spine_angle_address = calculate_spine_angle(key_frames['address']) if key_frames['address'] else None
        spine_angle_impact = calculate_spine_angle(key_frames['impact']) if key_frames['impact'] else None
        spine_angle_change = None
        if spine_angle_address is not None and spine_angle_impact is not None:
            spine_angle_change = round(spine_angle_impact - spine_angle_address, 1)
        
        # Log rotation values for debugging
        print("=== Rotation Calculation (Z-axis Depth - Frontal Plane) ===")
        print(f"Hip Z-values - Address: {baseline_hip_z}, Backswing: {backswing_hip_z}, Impact: {impact_hip_z}")
        print(f"Hip ROTATION (delta) - Backswing: {backswing_hip_rotation}°, Impact: {impact_hip_rotation}°")
        print(f"Shoulder Z-values - Address: {baseline_shoulder_z}, Backswing: {backswing_shoulder_z}, Impact: {impact_shoulder_z}")
        print(f"Shoulder ROTATION (delta) - Backswing: {backswing_shoulder_rotation}°, Impact: {impact_shoulder_rotation}°")
        print("============================================================")
        
        # Return analysis
        analysis = {
            'frames_analyzed': frame_count,
            'poses_detected': poses_detected,
            'duration_seconds': round(frame_count / fps, 2) if fps > 0 else 0,
            'fps': round(fps, 2),
            'detection_rate': round((poses_detected / frame_count * 100), 1) if frame_count > 0 else 0,
            'status': f'Pose detected in {poses_detected} of {frame_count} frames',
            # Rotation amounts (DELTA/CHANGE from address, not absolute angles)
            'hip_rotation_backswing': backswing_hip_rotation,  # Should be 30-60°
            'shoulder_rotation_backswing': backswing_shoulder_rotation,  # Should be 80-110°
            'hip_rotation_impact': impact_hip_rotation,  # Should be 30-45°
            'shoulder_rotation_impact': impact_shoulder_rotation,  # Should be near 0°
            # Alternative field names for compatibility
            'backswing_hip_rotation': backswing_hip_rotation,
            'backswing_shoulder_rotation': backswing_shoulder_rotation,
            'impact_hip_rotation': impact_hip_rotation,
            'impact_shoulder_rotation': impact_shoulder_rotation,
            # Legacy fields for backward compatibility (using ROTATION values, not absolute angles)
            'address_hip_angle': baseline_hip_z,  # This is the baseline Z-axis depth value
            'backswing_hip_angle': backswing_hip_rotation,  # This is ROTATION (delta)
            'impact_hip_angle': impact_hip_rotation,  # This is ROTATION (delta)
            'address_shoulder_angle': baseline_shoulder_z,  # This is the baseline Z-axis depth value
            'backswing_shoulder_angle': backswing_shoulder_rotation,  # This is ROTATION (delta)
            'impact_shoulder_angle': impact_shoulder_rotation,  # This is ROTATION (delta)
            # Spine angle measurements (unchanged)
            'spine_angle_address': spine_angle_address,
            'spine_angle_impact': spine_angle_impact,
            'spine_angle_change': spine_angle_change
        }
        
        return analysis, None
        
    except Exception as e:
        return None, f"Error analyzing video: {str(e)}"


def diagnose_swing(analysis_data):
    """Evaluate swing angles and identify issues based on research-backed ranges"""
    if not analysis_data:
        return None
    
    issues_detected = []
    strengths = []
    
    # HIP ROTATION (Backswing) - using rotation delta/change amount
    backswing_hip = (analysis_data.get('hip_rotation_backswing') or 
                     analysis_data.get('backswing_hip_rotation') or 
                     analysis_data.get('backswing_hip_angle'))
    if backswing_hip is not None:
        backswing_hip = round(float(backswing_hip), 1)
        if 30 <= backswing_hip <= 60:
            strengths.append("Excellent hip rotation in backswing")
        elif backswing_hip < 30:
            issues_detected.append({
                'category': 'Hip Rotation',
                'issue': 'Limited hip rotation in backswing',
                'description': f'Your hip rotation of {backswing_hip}° may benefit from improvement. Greater hip turn can help generate more power and maintain proper sequencing.',
                'severity': 'needs_attention'
            })
        elif backswing_hip > 60:
            issues_detected.append({
                'category': 'Hip Rotation',
                'issue': 'Excessive hip rotation in backswing',
                'description': f'Your hip rotation of {backswing_hip}° may be excessive. This can lead to loss of posture and inconsistent ball striking.',
                'severity': 'needs_attention'
            })
    
    # SHOULDER ROTATION (Backswing) - using rotation delta/change amount
    backswing_shoulder = (analysis_data.get('shoulder_rotation_backswing') or 
                          analysis_data.get('backswing_shoulder_rotation') or 
                          analysis_data.get('backswing_shoulder_angle'))
    if backswing_shoulder is not None:
        backswing_shoulder = round(float(backswing_shoulder), 1)
        if 80 <= backswing_shoulder <= 110:
            strengths.append("Good shoulder turn in backswing")
        elif backswing_shoulder < 80:
            issues_detected.append({
                'category': 'Shoulder Rotation',
                'issue': 'Limited shoulder rotation in backswing',
                'description': f'Your shoulder turn of {backswing_shoulder}° may benefit from improvement. A fuller shoulder turn is commonly associated with increased clubhead speed and better ball striking.',
                'severity': 'needs_attention'
            })
    
    # HIP-SHOULDER SEPARATION (X-Factor)
    if backswing_hip is not None and backswing_shoulder is not None:
        x_factor = round(abs(backswing_shoulder - backswing_hip), 1)
        if 40 <= x_factor <= 60:
            strengths.append("Good hip-shoulder separation (X-Factor)")
        elif x_factor < 30:
            issues_detected.append({
                'category': 'X-Factor',
                'issue': 'Limited hip-shoulder separation',
                'description': f'Your X-Factor of {x_factor}° may benefit from improvement. Greater separation between hip and shoulder turn can help create more power and maintain proper sequencing.',
                'severity': 'needs_attention'
            })
    
    # HIP ROTATION AT IMPACT - using rotation delta/change amount
    impact_hip = (analysis_data.get('hip_rotation_impact') or 
                  analysis_data.get('impact_hip_rotation') or 
                  analysis_data.get('impact_hip_angle'))
    if impact_hip is not None:
        impact_hip = round(float(impact_hip), 1)
        # At impact, hips should have rotated 30-45° from address
        if 30 <= impact_hip <= 45:
            strengths.append("Good hip rotation at impact")
        elif impact_hip < 30:
            issues_detected.append({
                'category': 'Impact Position',
                'issue': 'Limited hip rotation at impact',
                'description': f'Your hip rotation of {impact_hip}° at impact may benefit from improvement. Proper hip turn through impact is commonly associated with better power transfer and ball striking.',
                'severity': 'needs_attention'
            })
        elif impact_hip > 45:
            issues_detected.append({
                'category': 'Impact Position',
                'issue': 'Excessive hip rotation at impact',
                'description': f'Your hip rotation of {impact_hip}° at impact may be excessive. This can lead to inconsistent contact.',
                'severity': 'needs_attention'
            })
    
    # SHOULDER ROTATION AT IMPACT - should return close to address (0-10° difference)
    impact_shoulder = (analysis_data.get('shoulder_rotation_impact') or 
                       analysis_data.get('impact_shoulder_rotation') or 
                       analysis_data.get('impact_shoulder_angle'))
    if impact_shoulder is not None:
        impact_shoulder = round(float(impact_shoulder), 1)
        # At impact, shoulders should return close to address position
        if impact_shoulder <= 10:
            strengths.append("Good shoulder return at impact")
        elif impact_shoulder > 20:
            issues_detected.append({
                'category': 'Impact Position',
                'issue': 'Shoulders not returning to address position',
                'description': f'Your shoulder rotation of {impact_shoulder}° at impact suggests shoulders may not be returning properly. This can affect consistency.',
                'severity': 'needs_attention'
            })
    
    # SPINE ANGLE CHANGE
    spine_change = analysis_data.get('spine_angle_change')
    if spine_change is not None:
        spine_change = round(float(spine_change), 1)
        spine_change_abs = abs(spine_change)
        if spine_change_abs < 10:
            strengths.append("Maintained spine angle throughout swing")
        elif spine_change_abs > 15:
            issues_detected.append({
                'category': 'Spine Angle',
                'issue': 'Early extension detected',
                'description': f'Your spine angle change of {spine_change}° suggests early extension. This is commonly associated with inconsistent contact and loss of power. Maintaining spine angle through impact can improve ball striking.',
                'severity': 'needs_attention'
            })
    
    # Limit to maximum 2 issues, prioritize most impactful
    # Priority order: 1) Early extension (>15° spine change), 2) Restricted hip turn (<30°), 3) Restricted shoulder turn (<70°)
    priority_order = {
        'Spine Angle': 1,  # Highest priority - early extension
        'Hip Rotation': 2,  # Second priority - restricted hip turn
        'Shoulder Rotation': 3,  # Third priority - restricted shoulder turn
        'X-Factor': 4,
        'Impact Position': 5
    }
    
    # Sort by priority and limit to maximum 2 issues
    if len(issues_detected) > 2:
        issues_detected.sort(key=lambda x: priority_order.get(x['category'], 99))
        issues_detected = issues_detected[:2]
    elif len(issues_detected) > 0:
        # Sort even if 2 or fewer to ensure priority issue is first
        issues_detected.sort(key=lambda x: priority_order.get(x['category'], 99))
    
    # Determine priority issue
    priority_issue = None
    if issues_detected:
        priority_issue = issues_detected[0]
    
    # Create summary
    if not issues_detected and strengths:
        summary = "Your swing shows several positive characteristics. Continue focusing on maintaining these fundamentals."
    elif issues_detected and strengths:
        summary = f"Your swing has {len(strengths)} strong point(s) and {len(issues_detected)} area(s) that may benefit from focused practice."
    elif issues_detected:
        summary = f"Analysis identified {len(issues_detected)} area(s) that may benefit from improvement. Focused practice on these fundamentals can help enhance your swing."
    else:
        summary = "Analysis complete. Continue working on maintaining consistent fundamentals."
    
    diagnosis = {
        'issues_detected': issues_detected,
        'strengths': strengths,
        'summary': summary,
        'priority_issue': priority_issue
    }
    
    return diagnosis


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
            response_data['diagnosis'] = None
        else:
            response_data['analysis'] = analysis_data
            response_data['analysis_error'] = None
            # Generate diagnosis from analysis data
            diagnosis = diagnose_swing(analysis_data)
            response_data['diagnosis'] = diagnosis
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


@app.route('/api/signup', methods=['POST'])
def signup():
    """Create a new user account"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        confirm_password = data.get('confirm_password', '')
        
        # Validation
        if not email or not password:
            return jsonify({
                'error': 'Missing fields',
                'message': 'Email and password are required'
            }), 400
        
        if password != confirm_password:
            return jsonify({
                'error': 'Password mismatch',
                'message': 'Passwords do not match'
            }), 400
        
        if len(password) < 6:
            return jsonify({
                'error': 'Weak password',
                'message': 'Password must be at least 6 characters'
            }), 400
        
        # Create user
        user = create_user(email, password)
        
        if user is None:
            return jsonify({
                'error': 'Email exists',
                'message': 'An account with this email already exists'
            }), 409
        
        # Create session
        token = create_session(user['id'])
        
        return jsonify({
            'success': True,
            'message': 'Account created successfully',
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'subscription_status': user['subscription_status']
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


@app.route('/api/login', methods=['POST'])
def login():
    """Authenticate user and return session token"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'error': 'Missing fields',
                'message': 'Email and password are required'
            }), 400
        
        # Get user
        user = get_user_by_email(email)
        
        if user is None:
            return jsonify({
                'error': 'Invalid credentials',
                'message': 'Invalid email or password'
            }), 401
        
        # Verify password
        if not verify_password(user, password):
            return jsonify({
                'error': 'Invalid credentials',
                'message': 'Invalid email or password'
            }), 401
        
        # Create session
        token = create_session(user['id'])
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'subscription_status': user['subscription_status']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


@app.route('/api/logout', methods=['POST'])
def logout():
    """Clear user session"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if token:
            delete_session(token)
        
        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'message': str(e)
        }), 500


@app.route('/api/check-session', methods=['GET'])
def check_session():
    """Verify if user is logged in"""
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        
        if not token:
            return jsonify({
                'authenticated': False,
                'message': 'No token provided'
            }), 401
        
        user_id = verify_session(token)
        
        if user_id is None:
            return jsonify({
                'authenticated': False,
                'message': 'Invalid or expired session'
            }), 401
        
        # Get user info
        user = get_user_by_id(user_id)
        
        if user is None:
            return jsonify({
                'authenticated': False,
                'message': 'User not found'
            }), 401
        
        return jsonify({
            'authenticated': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'subscription_status': user['subscription_status']
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'authenticated': False,
            'error': str(e)
        }), 500
if __name__ == '__main__':
    init_database()
    app.run(debug=True, port=5000)
