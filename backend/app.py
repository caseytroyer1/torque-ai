from flask import Flask, request, jsonify
import os

os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ['EGL_PLATFORM'] = 'surfaceless'

import base64
import json
import math
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import numpy as np
from database import init_database, create_user, get_user_by_email, get_user_by_id, verify_password
from auth import create_session, verify_session, delete_session

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/analyze-swing', methods=['OPTIONS'])
def handle_options():
    response = app.make_default_options_response()
    return response

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


def _extract_and_annotate_frame(video_path, frame_index, landmarks, spine_angle, max_width=640):
    """Extract one frame from video, draw skeleton overlay, return base64 JPEG string."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    else:
        new_w, new_h = w, h

    def to_px(lm):
        return (int(lm.x * new_w), int(lm.y * new_h))

    BLUE_BGR = (255, 94, 30)   # #1E5EFF in BGR
    WHITE = (255, 255, 255)
    SPINE_COLOR = (255, 200, 0)  # bright blue-white for spine
    RADIUS = 5
    THICKNESS_LINE = 2

    if landmarks and len(landmarks) > 28:
        # Draw skeleton lines first (under joints)
        line_pairs = [(11, 12), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]
        for i, j in line_pairs:
            if i < len(landmarks) and j < len(landmarks):
                cv2.line(frame, to_px(landmarks[i]), to_px(landmarks[j]), WHITE, THICKNESS_LINE)

        # Spine line
        shoulder_mid = (int((landmarks[11].x + landmarks[12].x) / 2 * new_w), int((landmarks[11].y + landmarks[12].y) / 2 * new_h))
        hip_mid = (int((landmarks[23].x + landmarks[24].x) / 2 * new_w), int((landmarks[23].y + landmarks[24].y) / 2 * new_h))
        cv2.line(frame, hip_mid, shoulder_mid, SPINE_COLOR, 3)

        # Draw joints on top
        for lm in landmarks:
            cv2.circle(frame, to_px(lm), RADIUS, BLUE_BGR, -1)
            cv2.circle(frame, to_px(lm), RADIUS, WHITE, 1)

    _, buf = cv2.imencode('.jpg', frame)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def analyze_frames_with_claude(address_frame_b64, backswing_frame_b64, impact_frame_b64, mediapipe_data=None, golfer_hand='right', golfer_club='iron', user_camera_angle='behind'):
    """Send the three key frames to Claude Vision for golf swing analysis.
    MediaPipe measurements are computed deterministically; Claude fills in visual
    assessments and coaching notes around them. Returns parsed JSON or None."""
    if not address_frame_b64 or not backswing_frame_b64 or not impact_frame_b64:
        return None
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    md = mediapipe_data or {}

    # ------ MediaPipe-driven grades (deterministic) ------
    def _setup_grade_to_tier(g):
        if g == 'A':
            return 'Elite'
        if g == 'B':
            return 'Solid Foundation'
        return 'Needs Attention'

    setup_tier = _setup_grade_to_tier(md.get('setup_grade'))

    spine_change = md.get('spine_angle_change')
    spine_change_abs = abs(spine_change) if spine_change is not None else None
    tempo_ratio = md.get('tempo_ratio')

    # Backswing tier: spine maintained + tempo
    if spine_change_abs is not None and tempo_ratio is not None:
        if spine_change_abs < 10 and tempo_ratio >= 2.5:
            backswing_tier = 'Elite'
        elif spine_change_abs <= 15 and tempo_ratio >= 1.8:
            backswing_tier = 'Solid Foundation'
        else:
            backswing_tier = 'Needs Attention'
    else:
        backswing_tier = 'Solid Foundation'

    # Impact tier: spine change is primary driver
    if spine_change_abs is not None:
        if spine_change_abs < 10:
            impact_tier = 'Elite'
        elif spine_change_abs <= 15:
            impact_tier = 'Solid Foundation'
        else:
            impact_tier = 'Needs Attention'
    else:
        impact_tier = 'Solid Foundation'

    # Spine assessment string (address, qualitative from number)
    spine_addr = md.get('setup_spine_angle') or md.get('spine_angle_address')
    if spine_addr is not None:
        if 20 <= spine_addr <= 35:
            spine_assessment_str = 'good'
        elif 15 <= spine_addr < 20 or 35 < spine_addr <= 40:
            spine_assessment_str = 'slightly off'
        else:
            spine_assessment_str = 'needs work'
    else:
        spine_assessment_str = None

    # Knee flex qualitative
    knee_flex = md.get('setup_knee_flex')
    if knee_flex is not None:
        if 150 <= knee_flex <= 170:
            knee_flex_str = 'appropriate'
        elif knee_flex < 150:
            knee_flex_str = 'too bent'
        else:
            knee_flex_str = 'too straight'
    else:
        knee_flex_str = None

    # Shoulder level qualitative
    shoulder_level = md.get('setup_shoulder_level')
    if shoulder_level is not None:
        if shoulder_level <= 0.03:
            shoulder_level_str = 'level'
        elif shoulder_level <= 0.05:
            shoulder_level_str = 'slightly uneven'
        else:
            shoulder_level_str = 'uneven'
    else:
        shoulder_level_str = None

    # Spine maintained qualitative
    if spine_change_abs is not None:
        if spine_change_abs < 10:
            spine_maintained_str = 'yes'
        elif spine_change_abs <= 15:
            spine_maintained_str = 'slightly lost'
        else:
            spine_maintained_str = 'significantly lost'
    else:
        spine_maintained_str = None

    # Build a compact MediaPipe context string for Claude
    mp_facts = []
    if spine_addr is not None:
        mp_facts.append(f"- Spine angle at address: {spine_addr}° (assessment: {spine_assessment_str})")
    if knee_flex is not None:
        mp_facts.append(f"- Knee flex: {knee_flex}° ({knee_flex_str})")
    if shoulder_level is not None:
        mp_facts.append(f"- Shoulder level delta: {shoulder_level} ({shoulder_level_str})")
    if spine_change is not None:
        mp_facts.append(f"- Spine angle change address->impact: {spine_change}° ({spine_maintained_str})")
    if tempo_ratio is not None:
        mp_facts.append(f"- Tempo ratio (backswing:downswing): {tempo_ratio}:1")
    if md.get('setup_grade'):
        mp_facts.append(f"- Setup grade (MediaPipe): {md.get('setup_grade')} -> tier: {setup_tier}")
    mp_facts.append(f"- Backswing tier (computed): {backswing_tier}")
    mp_facts.append(f"- Impact tier (computed): {impact_tier}")
    mp_context = "\n".join(mp_facts) if mp_facts else "No MediaPipe measurements available."

    club_desc = (
        'DRIVER — expect a wider stance, more spine tilt away from target, and a shallower swing arc' if golfer_club == 'driver'
        else 'FAIRWAY WOOD — expect a slightly narrower stance than driver with a shallow to medium swing arc' if golfer_club == 'fairway'
        else 'WEDGE — expect a narrow stance, upright posture, and a shorter controlled backswing' if golfer_club == 'wedge'
        else 'IRON — expect a standard athletic stance with a medium swing arc'
    )

    # Angle-specific prompt JSON — only ask Claude to assess what is actually visible from this angle
    if user_camera_angle == 'face_on':
        prompt_json = '''{
  "address": {
    "posture": "good/needs work",
    "shoulder_level": "level/slightly uneven/uneven",
    "weight_distribution": "balanced/too much on left/too much on right",
    "coaching_note": "one specific actionable tip about setup in 15 words or less"
  },
  "backswing": {
    "lateral_head_movement": "steady/swayed left/swayed right",
    "weight_transfer": "good/minimal/reverse pivot",
    "coaching_note": "one specific actionable tip about backswing in 15 words or less"
  },
  "impact": {
    "shoulder_tilt": "good/too flat/too steep",
    "head_position": "steady/moved forward/moved back",
    "weight_transfer": "good/minimal/reversed",
    "coaching_note": "one specific actionable tip about impact in 15 words or less"
  },
  "overall": {
    "biggest_strength": "one sentence about the best part of this swing",
    "primary_focus": "the single most important thing to work on",
    "summary": "2-3 sentence overall assessment like a real golf coach would give"
  }
}'''
    elif user_camera_angle == 'down_the_line':
        prompt_json = '''{
  "address": {
    "posture": "good/needs work",
    "weight_distribution": "balanced/too much on heels/too much on toes",
    "coaching_note": "one specific actionable tip about setup in 15 words or less"
  },
  "backswing": {
    "left_arm": "straight/slightly bent/too bent",
    "weight_transfer": "good/minimal/reverse pivot",
    "coaching_note": "one specific actionable tip about backswing in 15 words or less"
  },
  "impact": {
    "hip_clearance": "good/needs more/excessive",
    "shoulder_position": "square/open/closed",
    "head_position": "steady/moved forward/moved back",
    "weight_transfer": "good/minimal/reversed",
    "coaching_note": "one specific actionable tip about impact in 15 words or less"
  },
  "overall": {
    "biggest_strength": "one sentence about the best part of this swing",
    "primary_focus": "the single most important thing to work on",
    "summary": "2-3 sentence overall assessment like a real golf coach would give"
  }
}'''
    else:
        # behind (default)
        prompt_json = '''{
  "address": {
    "posture": "good/needs work",
    "weight_distribution": "balanced/too much on heels/too much on toes",
    "coaching_note": "one specific actionable tip about setup in 15 words or less"
  },
  "backswing": {
    "left_arm": "straight/slightly bent/too bent",
    "weight_transfer": "good/minimal/reverse pivot",
    "coaching_note": "one specific actionable tip about backswing in 15 words or less"
  },
  "impact": {
    "head_position": "steady/moved forward/moved back",
    "weight_transfer": "good/minimal/reversed",
    "coaching_note": "one specific actionable tip about impact in 15 words or less"
  },
  "overall": {
    "biggest_strength": "one sentence about the best part of this swing",
    "primary_focus": "the single most important thing to work on",
    "summary": "2-3 sentence overall assessment like a real golf coach would give"
  }
}'''

    angle_desc = (
        'FACE-ON (camera is directly in front of the golfer, facing them). '
        'From this angle you can assess: shoulder level, lateral head movement, weight shift left/right, shoulder tilt at impact. '
        'You CANNOT reliably assess left arm straightness, hip/shoulder separation, or heel/toe weight distribution from this angle.'
        if user_camera_angle == 'face_on'
        else 'DOWN-THE-LINE (camera is behind and to the side, looking down the target line). '
        'From this angle you can assess: left arm straightness, heel/toe weight distribution, hip clearance, shoulder position at impact. '
        'You CANNOT reliably assess lateral head movement or left/right weight shift from this angle.'
        if user_camera_angle == 'down_the_line'
        else 'BEHIND (camera is directly behind the golfer). '
        'From this angle you can assess: posture, head position, and general weight distribution. '
        'You CANNOT reliably assess left arm straightness, hip clearance, or lateral weight shift from this angle.'
    )

    instruction_text = (
        f"You are an expert PGA golf instructor analyzing a golf swing. The golfer is "
        f"{'LEFT' if golfer_hand == 'left' else 'RIGHT'}-handed and is hitting a {club_desc}. "
        f"Camera angle: {angle_desc}\n\n"
        f"I am sending you 3 key frames from a golf swing video: Frame 1 is ADDRESS, Frame 2 is "
        f"BACKSWING TOP, Frame 3 is IMPACT.\n\n"
        f"MEDIAPIPE MEASUREMENTS (these are authoritative — do not contradict them):\n"
        f"{mp_context}\n\n"
        f"Your job is to provide ONLY the visual assessments that MediaPipe cannot measure, "
        f"limited to what is actually visible from the {user_camera_angle.replace('_', ' ')} angle. "
        f"Write coaching notes in second person — speak directly to the golfer using 'you' and 'your'. "
        f"Reference the measurements when relevant. Return ONLY a JSON object with no extra text, no markdown, "
        f"no code blocks. Use exactly this format:"
    )

    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction_text},
                    {"type": "text", "text": prompt_json},
                    {"type": "text", "text": "Frame 1 - ADDRESS:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": address_frame_b64}},
                    {"type": "text", "text": "Frame 2 - BACKSWING TOP:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": backswing_frame_b64}},
                    {"type": "text", "text": "Frame 3 - IMPACT:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": impact_frame_b64}}
                ]
            }
        ]
    )
    response_text = message.content[0].text.strip()
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
    response_text = response_text.strip()
    claude_analysis = json.loads(response_text)

    # ------ Inject MediaPipe-driven fields into Claude's response ------
    # These fields were previously guessed by Claude; now they come from MediaPipe.
    if 'address' not in claude_analysis or not isinstance(claude_analysis['address'], dict):
        claude_analysis['address'] = {}
    if 'backswing' not in claude_analysis or not isinstance(claude_analysis['backswing'], dict):
        claude_analysis['backswing'] = {}
    if 'impact' not in claude_analysis or not isinstance(claude_analysis['impact'], dict):
        claude_analysis['impact'] = {}

    if spine_addr is not None:
        claude_analysis['address']['spine_angle'] = spine_addr
    if spine_assessment_str is not None:
        claude_analysis['address']['spine_assessment'] = spine_assessment_str
    if knee_flex is not None:
        claude_analysis['address']['knee_flex_degrees'] = knee_flex
    if knee_flex_str is not None:
        claude_analysis['address']['knee_flex'] = knee_flex_str
    if shoulder_level_str is not None:
        claude_analysis['address']['shoulder_level'] = shoulder_level_str
    claude_analysis['address']['overall_setup'] = setup_tier

    if spine_maintained_str is not None:
        claude_analysis['backswing']['spine_angle_maintained'] = spine_maintained_str
    claude_analysis['backswing']['overall_backswing'] = backswing_tier

    if spine_maintained_str is not None:
        claude_analysis['impact']['spine_angle_maintained'] = spine_maintained_str
    claude_analysis['impact']['overall_impact'] = impact_tier

    return claude_analysis


def detect_camera_angle(landmarks):
    """
    Classify camera as face-on vs down-the-line using pose landmarks.
    Primary: shoulder width (X spread) / torso height.
    Secondary: nose lateral offset from hip midpoint, normalized by shoulder width.
    """
    NOSE = 0
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24
    default = {
        'angle': 'unknown',
        'confidence': 'low',
        'shoulder_width_ratio': 0.0,
        'nose_offset_normalized': 0.0,
        'message': 'Insufficient pose landmarks for camera angle.',
    }
    if not landmarks or len(landmarks) <= max(LEFT_HIP, RIGHT_HIP, RIGHT_SHOULDER, NOSE):
        return default

    try:
        ls = landmarks[LEFT_SHOULDER]
        rs = landmarks[RIGHT_SHOULDER]
        lh = landmarks[LEFT_HIP]
        rh = landmarks[RIGHT_HIP]
        nose = landmarks[NOSE]
    except (IndexError, TypeError, AttributeError):
        return default

    shoulder_width = abs(rs.x - ls.x)
    shoulder_mid_y = (ls.y + rs.y) / 2
    hip_mid_x = (lh.x + rh.x) / 2
    hip_mid_y = (lh.y + rh.y) / 2
    torso_height = max(abs(hip_mid_y - shoulder_mid_y), 1e-6)
    shoulder_width_ratio = shoulder_width / torso_height

    nose_offset_lateral = nose.x - hip_mid_x
    nose_offset_normalized = nose_offset_lateral / max(shoulder_width, 1e-6)

    if shoulder_width_ratio > 0.55:
        angle = 'face_on'
    elif shoulder_width_ratio < 0.35:
        angle = 'down_the_line'
    else:
        angle = 'unknown'

    # Confidence: primary clarity + secondary (nose vs hips alignment)
    abs_nose = abs(nose_offset_normalized)
    if angle == 'face_on':
        if abs_nose < 0.3:
            confidence = 'high'
        elif abs_nose < 0.55:
            confidence = 'medium'
        else:
            confidence = 'low'
    elif angle == 'down_the_line':
        if abs_nose > 0.12:
            confidence = 'high'
        elif abs_nose > 0.06:
            confidence = 'medium'
        else:
            confidence = 'low'
    else:
        confidence = 'low'

    message = (
        f"Shoulder width / torso height = {shoulder_width_ratio:.2f} "
        f"(face-on if >0.55, down-the-line if <0.35); "
        f"nose offset (normalized) = {nose_offset_normalized:.2f}. "
        f"Estimated: {angle} ({confidence} confidence)."
    )
    return {
        'angle': angle,
        'confidence': confidence,
        'shoulder_width_ratio': float(round(shoulder_width_ratio, 4)),
        'nose_offset_normalized': float(round(nose_offset_normalized, 4)),
        'message': message,
    }


def calculate_hip_z_rotation(landmarks):
    """Calculate hip rotation angle in degrees using atan2"""
    LEFT_HIP, RIGHT_HIP = 23, 24
    if not landmarks or len(landmarks) <= max(LEFT_HIP, RIGHT_HIP):
        return None
    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]
    angle = math.degrees(math.atan2(right_hip.z - left_hip.z, right_hip.x - left_hip.x))
    return angle


def calculate_shoulder_z_rotation(landmarks):
    """Calculate shoulder rotation angle in degrees using atan2"""
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    if not landmarks or len(landmarks) <= max(LEFT_SHOULDER, RIGHT_SHOULDER):
        return None
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    angle = math.degrees(math.atan2(right_shoulder.z - left_shoulder.z, right_shoulder.x - left_shoulder.x))
    return angle


def calculate_rotation_degrees(world_landmarks_a, world_landmarks_b, joint='hip'):
    """
    Rotation in degrees between two poses in the horizontal (X–Z) plane.
    Uses vector from right to left hip (or shoulder); heading = atan2(vx, vz) per frame;
    returns smallest angle between headings in [0, 180].
    """
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    if joint == 'hip':
        li, ri = LEFT_HIP, RIGHT_HIP
    else:
        li, ri = LEFT_SHOULDER, RIGHT_SHOULDER
    if not world_landmarks_a or not world_landmarks_b:
        return None
    try:
        n = max(li, ri) + 1
        if len(world_landmarks_a) < n or len(world_landmarks_b) < n:
            return None
    except (TypeError, AttributeError):
        return None

    def heading(wl):
        left = wl[li]
        right = wl[ri]
        vx = left.x - right.x
        vz = left.z - right.z
        mag = math.sqrt(vx * vx + vz * vz)
        if mag < 1e-9:
            return None
        return math.degrees(math.atan2(vx, vz))

    h1 = heading(world_landmarks_a)
    h2 = heading(world_landmarks_b)
    if h1 is None or h2 is None:
        return None
    diff = abs(h1 - h2)
    diff = min(diff, 360.0 - diff)
    return round(float(diff), 1)


def analyze_video_with_mediapipe(video_path, golfer_hand='right', golfer_club='iron', user_camera_angle='behind'):
    """Analyze golf swing video using MediaPipe Pose detection"""
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import cv2
        import math
        
        # Set up MediaPipe Pose Landmarker
        base_options = python.BaseOptions(model_asset_path=os.path.join(os.path.dirname(__file__), 'pose_landmarker.task'))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        
        detector = vision.PoseLandmarker.create_from_options(options)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Could not open video file"
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] total_frames={total_frames}, fps={fps}")
        RIGHT_WRIST = 16
        
        # STEP 1: First pass - collect pose, right wrist Y, and world landmarks per frame
        # frame_data[i] = (landmarks, wrist_y, world_landmarks) or None — always use helpers below
        frame_data = [None] * total_frames
        poses_detected = 0
        frames_with_world = 0

        def _entry_landmarks(entry):
            if not entry:
                return None
            return entry[0] if len(entry) >= 1 else None

        def _entry_wrist_y(entry):
            if not entry or len(entry) < 2:
                return None
            return entry[1]

        def _entry_world(entry):
            if not entry or len(entry) < 3:
                return None
            return entry[2]

        frames_read = 0
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_count / fps) * 1000) if fps > 0 else frame_count * 33
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                poses_detected += 1
                landmarks = detection_result.pose_landmarks[0]
                world_landmarks = None
                try:
                    pwl = detection_result.pose_world_landmarks
                except AttributeError:
                    pwl = None
                if pwl is not None and len(pwl) > 0:
                    world_landmarks = pwl[0]
                if world_landmarks is not None:
                    frames_with_world += 1
                if len(landmarks) > RIGHT_WRIST:
                    wrist_y = landmarks[RIGHT_WRIST].y
                    frame_data[frame_count] = (landmarks, wrist_y, world_landmarks)
                else:
                    frame_data[frame_count] = (landmarks, None, world_landmarks)
        
        cap.release()
        detector.close()

        print(
            f"[Pose pass] frames_read={frames_read} total_frames={total_frames} poses_detected={poses_detected} "
            f"frames_with_world={frames_with_world}"
        )
        
        # STEP 2: Find swing window and key frames intelligently
        MOTION_THRESHOLD = 0.015  # normalized Y change (tune for sensitivity)
        CONSECUTIVE_MOTION_FRAMES = 2  # require N frames above threshold
        ADDRESS_PRE_SWING_BUFFER = 30  # exclude this many frames before swing start from address search
        
        swing_start_frame = None
        address_frame_idx = None
        backswing_frame_idx = None
        impact_frame_idx = None
        address_wrist_y = None
        
        # Find first significant motion (swing start)
        for i in range(1, total_frames):
            if _entry_wrist_y(frame_data[i]) is None:
                continue
            if _entry_wrist_y(frame_data[i - 1]) is None:
                continue
            motion = abs(_entry_wrist_y(frame_data[i]) - _entry_wrist_y(frame_data[i - 1]))
            if motion <= MOTION_THRESHOLD:
                continue
            # Require CONSECUTIVE_MOTION_FRAMES above threshold
            count = 1
            j = i + 1
            while j < total_frames and count < CONSECUTIVE_MOTION_FRAMES:
                if _entry_wrist_y(frame_data[j]) is None or _entry_wrist_y(frame_data[j - 1]) is None:
                    break
                m = abs(_entry_wrist_y(frame_data[j]) - _entry_wrist_y(frame_data[j - 1]))
                if m > MOTION_THRESHOLD:
                    count += 1
                    j += 1
                else:
                    break
            if count >= CONSECUTIVE_MOTION_FRAMES:
                swing_start_frame = i
                print(f"[Swing detection] Swing start frame: {swing_start_frame} (motion threshold {MOTION_THRESHOLD}, first significant wrist movement)")
                break
        
        if swing_start_frame is not None:
            # ADDRESS: most stable wrist_y in early video; exclude last BUFFER frames before swing start when possible
            if swing_start_frame >= ADDRESS_PRE_SWING_BUFFER:
                window_start = 0
                window_end = swing_start_frame - ADDRESS_PRE_SWING_BUFFER
            else:
                window_start = 0
                window_end = swing_start_frame
            address_candidates = []
            for idx in range(window_start, window_end):
                wy = _entry_wrist_y(frame_data[idx])
                if wy is not None:
                    address_candidates.append((idx, wy))
            if address_candidates:
                # Pick frame whose wrist_y is closest to median (most typical "still" pose)
                wrist_ys = [y for _, y in address_candidates]
                median_y = sorted(wrist_ys)[len(wrist_ys) // 2]
                address_frame_idx = min(address_candidates, key=lambda c: abs(c[1] - median_y))[0]
                address_wrist_y = _entry_wrist_y(frame_data[address_frame_idx])
                print(f"[Address] Frame {address_frame_idx} (most stable of frames {window_start}-{window_end}, pre-swing buffer {ADDRESS_PRE_SWING_BUFFER}, wrist_y={address_wrist_y:.4f})")
            
            # BACKSWING TOP: right wrist at highest position (min Y) in swing window
            backswing_candidates = [
                (idx, _entry_wrist_y(frame_data[idx]))
                for idx in range(swing_start_frame, total_frames)
                if _entry_wrist_y(frame_data[idx]) is not None
            ]
            if backswing_candidates:
                backswing_frame_idx = min(backswing_candidates, key=lambda c: c[1])[0]
                print(f"[Backswing top] Frame {backswing_frame_idx} (right wrist at highest position, wrist_y={_entry_wrist_y(frame_data[backswing_frame_idx]):.4f})")
            
            # IMPACT: within downswing window only (avoid picking follow-through)
            # Window = backswing_frame to backswing_frame + (backswing - address) * 3
            if backswing_frame_idx is not None and address_frame_idx is not None and address_wrist_y is not None:
                downswing_length = int((backswing_frame_idx - address_frame_idx) * 3)
                impact_window_end_ideal = min(total_frames, backswing_frame_idx + 1 + downswing_length)

                # If fewer than 10 frames remain after backswing, the ideal window is
                # too short — search ALL remaining frames instead so late-swing videos
                # (golfer standing still for several seconds before swinging) still get
                # a valid impact frame.
                frames_after_backswing = total_frames - (backswing_frame_idx + 1)
                if frames_after_backswing < 15:
                    impact_window_end = total_frames
                    print(f"[Impact] Only {frames_after_backswing} frames after backswing — expanding impact search to end of video")
                else:
                    impact_window_end = impact_window_end_ideal

                impact_candidates = [
                    (idx, _entry_wrist_y(frame_data[idx]))
                    for idx in range(backswing_frame_idx + 1, impact_window_end)
                    if _entry_wrist_y(frame_data[idx]) is not None
                ]
                if impact_candidates:
                    impact_frame_idx = min(impact_candidates, key=lambda c: abs(c[1] - address_wrist_y))[0]
                    print(f"[Impact] Frame {impact_frame_idx} (window frames {backswing_frame_idx + 1}-{impact_window_end - 1}, wrist Y closest to address {address_wrist_y:.4f}, impact wrist_y={_entry_wrist_y(frame_data[impact_frame_idx]):.4f})")
        
        # STEP 3: Fallback to percentage-based frames if intelligent detection failed
        if address_frame_idx is None or backswing_frame_idx is None or impact_frame_idx is None:
            address_frame_idx = 0
            backswing_frame_idx = total_frames // 2
            impact_frame_idx = min(int(total_frames * 0.6), total_frames - 1) if total_frames > 0 else 0
            print(f"[Fallback] Using percentage-based frames: address={address_frame_idx}, backswing={backswing_frame_idx}, impact={impact_frame_idx}")
        
        # Build key_frames from normalized landmarks (tuple index 0) at selected indices
        def get_landmarks(idx):
            if idx is None or total_frames == 0 or idx < 0 or idx >= total_frames:
                return None
            return _entry_landmarks(frame_data[idx])

        def get_world_landmarks(idx):
            if idx is None or total_frames == 0 or idx < 0 or idx >= total_frames:
                return None
            return _entry_world(frame_data[idx])

        key_frames = {
            'address': get_landmarks(address_frame_idx),
            'backswing': get_landmarks(backswing_frame_idx),
            'impact': get_landmarks(impact_frame_idx)
        }
        print(
            f"[Swing diag] poses_detected={poses_detected} frames_with_world={frames_with_world} "
            f"address_frame_idx={address_frame_idx} backswing_frame_idx={backswing_frame_idx} "
            f"impact_frame_idx={impact_frame_idx}"
        )
        print(f"[Key frames] address={address_frame_idx}, backswing={backswing_frame_idx}, impact={impact_frame_idx}")

        cam_info = detect_camera_angle(key_frames['address'])
        print(f"[Camera angle auto-detected] {cam_info['message']}")

        # If user declared a specific angle, trust their selection over auto-detection
        # 'behind' and 'down_the_line' both use the same DTL measurement geometry
        if user_camera_angle in ('behind', 'down_the_line'):
            cam_info['angle'] = user_camera_angle
            cam_info['confidence'] = 'high'
            print(f"[Camera angle] User declared: {user_camera_angle} — overriding auto-detection")
        elif user_camera_angle == 'face_on':
            cam_info['angle'] = 'face_on'
            cam_info['confidence'] = 'high'
            print(f"[Camera angle] User declared: face_on — overriding auto-detection")

        # Calculate angles for key frames
        # MediaPipe pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        
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
        
        def angle_at_joint(landmarks, upper_idx, joint_idx, lower_idx):
            """Angle at joint (upper-joint-lower) in degrees. 180 = straight."""
            if not landmarks or len(landmarks) <= max(upper_idx, joint_idx, lower_idx):
                return None
            upper = landmarks[upper_idx]
            joint = landmarks[joint_idx]
            lower = landmarks[lower_idx]
            v1_x = upper.x - joint.x
            v1_y = upper.y - joint.y
            v2_x = lower.x - joint.x
            v2_y = lower.y - joint.y
            dot = v1_x * v2_x + v1_y * v2_y
            mag1 = math.sqrt(v1_x * v1_x + v1_y * v1_y)
            mag2 = math.sqrt(v2_x * v2_x + v2_y * v2_y)
            if mag1 < 1e-6 or mag2 < 1e-6:
                return None
            cos_a = max(-1, min(1, dot / (mag1 * mag2)))
            return round(math.degrees(math.acos(cos_a)), 1)
        
        # Hip/shoulder rotation (degrees) from world landmarks: address vs backswing / impact (X–Z plane)
        wl_address = get_world_landmarks(address_frame_idx)
        wl_backswing = get_world_landmarks(backswing_frame_idx)
        wl_impact = get_world_landmarks(impact_frame_idx)

        try:
            backswing_hip_rotation = calculate_rotation_degrees(wl_address, wl_backswing, 'hip')
            backswing_shoulder_rotation = calculate_rotation_degrees(wl_address, wl_backswing, 'shoulder')
            impact_hip_rotation = calculate_rotation_degrees(wl_address, wl_impact, 'hip')
            impact_shoulder_rotation = calculate_rotation_degrees(wl_address, wl_impact, 'shoulder')
        except Exception as rot_err:
            print(f"[Rotation world] Non-fatal error (using None): {rot_err}")
            backswing_hip_rotation = backswing_shoulder_rotation = impact_hip_rotation = impact_shoulder_rotation = None

        def _norm_rotation_delta(lm_addr, lm_target, joint):
            if not lm_addr or not lm_target:
                return None
            if joint == 'hip':
                h1, h2 = calculate_hip_z_rotation(lm_addr), calculate_hip_z_rotation(lm_target)
            else:
                h1, h2 = calculate_shoulder_z_rotation(lm_addr), calculate_shoulder_z_rotation(lm_target)
            if h1 is None or h2 is None:
                return None
            diff = abs(h1 - h2)
            diff = min(diff, 360.0 - diff)
            return round(float(diff), 1)

        if backswing_hip_rotation is None:
            backswing_hip_rotation = _norm_rotation_delta(key_frames['address'], key_frames['backswing'], 'hip')
        if backswing_shoulder_rotation is None:
            backswing_shoulder_rotation = _norm_rotation_delta(key_frames['address'], key_frames['backswing'], 'shoulder')
        if impact_hip_rotation is None:
            impact_hip_rotation = _norm_rotation_delta(key_frames['address'], key_frames['impact'], 'hip')
        if impact_shoulder_rotation is None:
            impact_shoulder_rotation = _norm_rotation_delta(key_frames['address'], key_frames['impact'], 'shoulder')

        print(
            f"[Rotation world] hip backswing={backswing_hip_rotation}° shoulder backswing={backswing_shoulder_rotation}° "
            f"hip impact={impact_hip_rotation}° shoulder impact={impact_shoulder_rotation}°"
        )

        # Calculate spine angle (unchanged - this is still a valid measurement)
        spine_angle_address = calculate_spine_angle(key_frames['address']) if key_frames['address'] else None
        spine_angle_impact = calculate_spine_angle(key_frames['impact']) if key_frames['impact'] else None
        spine_angle_change = None
        if spine_angle_address is not None and spine_angle_impact is not None:
            spine_angle_change = round(spine_angle_impact - spine_angle_address, 1)
        
        # Setup/posture metrics from address frame
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        setup_spine_angle = spine_angle_address
        setup_knee_flex_left = angle_at_joint(key_frames['address'], LEFT_HIP, LEFT_KNEE, LEFT_ANKLE) if key_frames['address'] else None
        setup_knee_flex_right = angle_at_joint(key_frames['address'], RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE) if key_frames['address'] else None
        setup_knee_flex = None
        if setup_knee_flex_left is not None and setup_knee_flex_right is not None:
            setup_knee_flex = round((setup_knee_flex_left + setup_knee_flex_right) / 2, 1)
        elif setup_knee_flex_left is not None:
            setup_knee_flex = setup_knee_flex_left
        elif setup_knee_flex_right is not None:
            setup_knee_flex = setup_knee_flex_right
        setup_shoulder_level = None
        if key_frames['address'] and len(key_frames['address']) > max(LEFT_SHOULDER, RIGHT_SHOULDER):
            left_y = key_frames['address'][LEFT_SHOULDER].y
            right_y = key_frames['address'][RIGHT_SHOULDER].y
            setup_shoulder_level = round(abs(left_y - right_y), 4)
        setup_hip_hinge = None
        if key_frames['address'] and len(key_frames['address']) > max(LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE):
            hip_mid_x = (key_frames['address'][LEFT_HIP].x + key_frames['address'][RIGHT_HIP].x) / 2
            ankle_mid_x = (key_frames['address'][LEFT_ANKLE].x + key_frames['address'][RIGHT_ANKLE].x) / 2
            setup_hip_hinge = round(hip_mid_x - ankle_mid_x, 4)
        # Setup grade: spine 20-35°, knee 150-170°, shoulder < 0.03, hip hinge in range (-0.08 to 0.08)
        in_range = 0
        if setup_spine_angle is not None and 20 <= setup_spine_angle <= 35:
            in_range += 1
        if setup_knee_flex is not None and 150 <= setup_knee_flex <= 170:
            in_range += 1
        if setup_shoulder_level is not None and setup_shoulder_level < 0.03:
            in_range += 1
        if setup_hip_hinge is not None and -0.08 <= setup_hip_hinge <= 0.08:
            in_range += 1
        if in_range == 4:
            setup_grade = 'A'
        elif in_range == 3:
            setup_grade = 'B'
        elif in_range == 2:
            setup_grade = 'C'
        else:
            setup_grade = 'D'
        
        # Return analysis (use frames_read — loop var frame_count is last index, not count)
        analysis = {
            'frames_analyzed': frames_read,
            'poses_detected': poses_detected,
            'duration_seconds': round(frames_read / fps, 2) if fps > 0 else 0,
            'fps': round(fps, 2),
            'detection_rate': round((poses_detected / frames_read * 100), 1) if frames_read > 0 else 0,
            'status': f'Pose detected in {poses_detected} of {frames_read} frames',
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
            # Legacy fields: rotation deltas are degrees (world X–Z); address_* no longer Z-proxy
            'address_hip_angle': None,
            'backswing_hip_angle': backswing_hip_rotation,
            'impact_hip_angle': impact_hip_rotation,
            'address_shoulder_angle': None,
            'backswing_shoulder_angle': backswing_shoulder_rotation,
            'impact_shoulder_angle': impact_shoulder_rotation,
            # Spine angle measurements (unchanged)
            'spine_angle_address': spine_angle_address,
            'spine_angle_impact': spine_angle_impact,
            'spine_angle_change': spine_angle_change,
            # Setup/posture at address
            'setup_spine_angle': setup_spine_angle,
            'setup_knee_flex': setup_knee_flex,
            'setup_knee_flex_left': setup_knee_flex_left,
            'setup_knee_flex_right': setup_knee_flex_right,
            'setup_shoulder_level': setup_shoulder_level,
            'setup_hip_hinge': setup_hip_hinge,
            'setup_grade': setup_grade,
            'golfer_hand': golfer_hand,
            'golfer_club': golfer_club,
            'camera_angle': cam_info['angle'],
            'camera_angle_confidence': cam_info['confidence'],
            'camera_angle_message': cam_info['message'],
            'shoulder_width_ratio': cam_info['shoulder_width_ratio'],
            'user_declared_camera_angle': user_camera_angle,
        }

        # SWING TEMPO CALCULATION
        # Tempo ratio = backswing duration / downswing duration (tour average ~3.0)
        tempo_ratio = None
        backswing_duration_sec = None
        downswing_duration_sec = None
        tempo_label = None

        if (address_frame_idx is not None and
            backswing_frame_idx is not None and
            impact_frame_idx is not None and
            fps and fps > 0 and
            backswing_frame_idx > address_frame_idx and
            impact_frame_idx > backswing_frame_idx):

            backswing_frames = backswing_frame_idx - address_frame_idx
            downswing_frames = impact_frame_idx - backswing_frame_idx

            if downswing_frames > 0:
                backswing_duration_sec = round(backswing_frames / fps, 2)
                downswing_duration_sec = round(downswing_frames / fps, 2)
                tempo_ratio = round(backswing_duration_sec / downswing_duration_sec, 1)

                if tempo_ratio >= 2.5:
                    tempo_label = "Tour Tempo"
                elif tempo_ratio >= 1.8:
                    tempo_label = "Good Tempo"
                elif tempo_ratio >= 1.2:
                    tempo_label = "Slightly Rushed"
                else:
                    tempo_label = "Too Fast — Rushing Downswing"

        analysis['tempo_ratio'] = tempo_ratio
        analysis['backswing_duration_sec'] = backswing_duration_sec
        analysis['downswing_duration_sec'] = downswing_duration_sec
        analysis['tempo_label'] = tempo_label

        # Extract key frames as images with skeleton overlay and add to response
        spine_angle_backswing = calculate_spine_angle(key_frames['backswing']) if key_frames['backswing'] else None
        print(f"[Frame extraction] address_frame={address_frame_idx}, backswing_frame={backswing_frame_idx}, impact_frame={impact_frame_idx}")
        try:
            # address_frame_image from frame index address_frame_idx
            analysis['address_frame_image'] = _extract_and_annotate_frame(
                video_path, address_frame_idx, key_frames['address'], spine_angle_address
            )
            # backswing_frame_image from frame index backswing_frame_idx
            analysis['backswing_frame_image'] = _extract_and_annotate_frame(
                video_path, backswing_frame_idx, key_frames['backswing'], spine_angle_backswing
            )
            # impact_frame_image from frame index impact_frame_idx
            analysis['impact_frame_image'] = _extract_and_annotate_frame(
                video_path, impact_frame_idx, key_frames['impact'], spine_angle_impact
            )
        except Exception as frame_err:
            print(f"[Frame extraction] Error: {frame_err}")
            analysis['address_frame_image'] = None
            analysis['backswing_frame_image'] = None
            analysis['impact_frame_image'] = None

        try:
            if analysis.get('address_frame_image') and analysis.get('backswing_frame_image') and analysis.get('impact_frame_image'):
                claude_analysis = analyze_frames_with_claude(
                    analysis['address_frame_image'],
                    analysis['backswing_frame_image'],
                    analysis['impact_frame_image'],
                    mediapipe_data=analysis,
                    golfer_hand=golfer_hand,
                    golfer_club=golfer_club,
                    user_camera_angle=user_camera_angle
                )
                analysis['claude_vision_analysis'] = claude_analysis
                if claude_analysis is not None:
                    print("Claude Vision analysis complete")
            else:
                analysis['claude_vision_analysis'] = None
        except Exception as e:
            print(f"Claude Vision error: {str(e)}")
            analysis['claude_vision_analysis'] = None

        return analysis, None
        
    except Exception as e:
        return None, f"Error analyzing video: {str(e)}"


def diagnose_swing(analysis_data):
    """Evaluate swing angles and identify issues based on research-backed ranges"""
    if not analysis_data:
        return None
    
    issues_detected = []
    strengths = []
    
    # SETUP / POSTURE (address)
    setup_knee_flex = analysis_data.get('setup_knee_flex')
    if setup_knee_flex is not None:
        setup_knee_flex = round(float(setup_knee_flex), 1)
        if 150 <= setup_knee_flex <= 170:
            strengths.append("Good knee flex at address")
        elif setup_knee_flex < 150:
            issues_detected.append({
                'category': 'Setup',
                'issue': 'Excessive knee flex',
                'description': f'Your knee flex of {setup_knee_flex}° at address may be too bent. A slight flex (150-170°) helps with balance and rotation.',
                'severity': 'needs_attention'
            })
        elif setup_knee_flex > 170:
            issues_detected.append({
                'category': 'Setup',
                'issue': 'Locked knees at address',
                'description': f'Your knee angle of {setup_knee_flex}° suggests knees may be too straight at address. A slight flex (150-170°) improves stability.',
                'severity': 'needs_attention'
            })
    setup_shoulder_level = analysis_data.get('setup_shoulder_level')
    if setup_shoulder_level is not None:
        setup_shoulder_level = float(setup_shoulder_level)
        if setup_shoulder_level <= 0.03:
            strengths.append("Level shoulders at address")
        elif setup_shoulder_level > 0.05:
            issues_detected.append({
                'category': 'Setup',
                'issue': 'Uneven shoulders at address',
                'description': f'Your shoulder height difference ({setup_shoulder_level:.3f}) suggests uneven setup. Level shoulders promote consistent ball striking.',
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

    # SWING TEMPO
    tempo_ratio = analysis_data.get('tempo_ratio')
    tempo_label = analysis_data.get('tempo_label')
    backswing_sec = analysis_data.get('backswing_duration_sec')
    downswing_sec = analysis_data.get('downswing_duration_sec')

    if tempo_ratio is not None and tempo_label is not None:
        if tempo_ratio >= 2.5:
            strengths.append(f"Excellent swing tempo ({tempo_ratio}:1 ratio — tour average is 3:1)")
        elif tempo_ratio >= 1.8:
            strengths.append(f"Good swing tempo ({tempo_ratio}:1 ratio — keep working toward 3:1)")
        elif tempo_ratio < 1.8:
            issues_detected.append({
                'category': 'Tempo',
                'issue': 'Rushing the downswing',
                'description': (
                    f"Your swing tempo ratio is {tempo_ratio}:1 "
                    f"(backswing {backswing_sec}s / downswing {downswing_sec}s). "
                    f"Tour pros average 3:1 — meaning the backswing takes 3x longer than the downswing. "
                    f"Try counting 'one-and-two' on your backswing and 'three' at impact to slow down and load properly."
                ),
                'severity': 'needs_attention'
            })

    # Limit to maximum 2 issues, prioritize most impactful
    # Priority order: 1) Early extension, 2) Setup, 3) Hip rotation, 4) Shoulder rotation, etc.
    priority_order = {
        'Tempo': 1,
        'Spine Angle': 2,
        'Setup': 3,
        'Hip Rotation': 4,
        'Shoulder Rotation': 5,
        'X-Factor': 6,
        'Impact Position': 7
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

    camera_angle = analysis_data.get('camera_angle') or 'unknown'
    camera_angle_confidence = (analysis_data.get('camera_angle_confidence') or 'low')
    if str(camera_angle_confidence).lower() not in ('high', 'medium', 'low'):
        camera_angle_confidence = 'low'
    cam_note = (
        " Note: camera angle may affect rotation accuracy — for best results, film face-on or down-the-line."
    )
    if (camera_angle == 'unknown' or camera_angle_confidence == 'low') and issues_detected:
        for issue in issues_detected:
            cat = issue.get('category', '')
            iss = (issue.get('issue') or '').lower()
            if cat in ('Hip Rotation', 'Shoulder Rotation'):
                issue['description'] = issue['description'].rstrip() + cam_note
            elif cat == 'Impact Position' and ('hip' in iss or 'shoulder' in iss):
                issue['description'] = issue['description'].rstrip() + cam_note

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
        golfer_hand = request.form.get('hand', 'right')
        golfer_club = request.form.get('club', 'iron')
        user_camera_angle = request.form.get('camera_angle', 'behind')
        
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
        analysis_data, analysis_error = analyze_video_with_mediapipe(file_path, golfer_hand=golfer_hand, golfer_club=golfer_club, user_camera_angle=user_camera_angle)
        
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


@app.route('/golf-coach', methods=['POST'])
def golf_coach():
    try:
        import anthropic
        data = request.get_json()
        user_message = data.get('message', '')
        swing_data = data.get('swing_data', None)
        history = data.get('history', [])

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return jsonify({'error': 'No API key found', 'success': False}), 500

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = """You are an expert PGA golf coach assistant for Torque AI. Speak directly to the golfer using 'you' and 'your' — never say 'this golfer' or 'the golfer'. Be conversational, encouraging, and direct like a real coach talking to a student on the range. Keep responses under 120 words. No headers — just natural coaching conversation."""

        if swing_data:
            parts = []
            if swing_data.get('club'): parts.append(f"Club: {swing_data['club']}")
            if swing_data.get('hand'): parts.append(f"Handed: {swing_data['hand']}")
            if swing_data.get('overall_summary'): parts.append(f"Swing summary: {swing_data['overall_summary']}")
            if swing_data.get('biggest_strength'): parts.append(f"Biggest strength: {swing_data['biggest_strength']}")
            if swing_data.get('primary_focus'): parts.append(f"Primary focus: {swing_data['primary_focus']}")
            if swing_data.get('primary_issue') and swing_data['primary_issue'] != 'None detected': parts.append(f"Main issue: {swing_data['primary_issue']}")
            if swing_data.get('issues') and swing_data['issues'] != 'None': parts.append(f"Issues detected: {swing_data['issues']}")
            if swing_data.get('strengths') and swing_data['strengths'] != 'None': parts.append(f"Strengths: {swing_data['strengths']}")
            if swing_data.get('spine_angle'): parts.append(f"Spine angle: {swing_data['spine_angle']}°")
            if swing_data.get('spine_maintained'): parts.append(f"Spine maintained: {swing_data['spine_maintained']}")
            if swing_data.get('tempo_ratio'): parts.append(f"Tempo: {swing_data['tempo_ratio']}:1 ({swing_data.get('tempo_label', '')})")
            if swing_data.get('hip_shoulder_separation'): parts.append(f"Hip/shoulder separation: {swing_data['hip_shoulder_separation']}")
            if swing_data.get('left_arm'): parts.append(f"Left arm: {swing_data['left_arm']}")
            if swing_data.get('weight_transfer'): parts.append(f"Weight transfer: {swing_data['weight_transfer']}")
            if swing_data.get('posture'): parts.append(f"Posture: {swing_data['posture']}")
            if parts:
                system_prompt += "\n\nGolfer's latest swing data:\n" + "\n".join(parts)

        messages = []
        for h in history[:-1]:
            role = h.get('role', 'user')
            if role in ('user', 'assistant'):
                messages.append({"role": role, "content": h.get('content', '')})
        messages.append({"role": "user", "content": user_message})

        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system_prompt,
            messages=messages
        )

        return jsonify({
            'response': message.content[0].text,
            'success': True
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"GOLF COACH ERROR: {str(e)}")
        print(f"TRACEBACK: {error_details}")
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    init_database()
    app.run(debug=True, port=5000)
