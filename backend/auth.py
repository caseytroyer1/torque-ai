import hashlib
import secrets
import time
from datetime import datetime, timedelta

# In-memory session storage (in production, use Redis or database)
sessions = {}

def hash_password(password):
    """Hash password using bcrypt-like approach (using hashlib for simplicity)"""
    # Using SHA-256 with salt for now (in production, use bcrypt)
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"

def verify_password(password, password_hash):
    """Verify password against hash"""
    try:
        salt, stored_hash = password_hash.split(':')
        password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return password_hash_check.hex() == stored_hash
    except:
        return False

def create_session(user_id):
    """Create a new session token for user"""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        'user_id': user_id,
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(days=7)  # 7 day expiration
    }
    return token

def verify_session(token):
    """Verify session token and return user_id if valid"""
    if not token or token not in sessions:
        return None
    
    session = sessions[token]
    
    # Check if session expired
    if datetime.now() > session['expires_at']:
        del sessions[token]
        return None
    
    return session['user_id']

def delete_session(token):
    """Delete a session token"""
    if token in sessions:
        del sessions[token]
        return True
    return False

def cleanup_expired_sessions():
    """Remove expired sessions (call periodically)"""
    current_time = datetime.now()
    expired_tokens = [
        token for token, session in sessions.items()
        if current_time > session['expires_at']
    ]
    for token in expired_tokens:
        del sessions[token]

