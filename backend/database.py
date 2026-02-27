import sqlite3
import json
from datetime import datetime
import os
from auth import hash_password, verify_password as auth_verify_password

# Database file path
DB_PATH = 'backend/swings.db'

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_database():
    """Initialize database and create tables if they don't exist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            subscription_status TEXT DEFAULT 'free'
        )
    ''')
    
    # Create Swings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS swings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            video_filename TEXT NOT NULL,
            analysis_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Create index on user_id for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_swings_user_id ON swings(user_id)
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def create_user(email, password):
    """Create a new user account"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (email, password_hash)
            VALUES (?, ?)
        ''', (email, password_hash))
        
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {'id': user_id, 'email': email, 'subscription_status': 'free'}
    except sqlite3.IntegrityError:
        conn.close()
        return None  # Email already exists
    except Exception as e:
        conn.close()
        raise e

def get_user_by_email(email):
    """Get user by email address"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, email, password_hash, created_at, subscription_status
        FROM users
        WHERE email = ?
    ''', (email,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'email': row['email'],
            'password_hash': row['password_hash'],
            'created_at': row['created_at'],
            'subscription_status': row['subscription_status']
        }
    return None

def get_user_by_id(user_id):
    """Get user by ID"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, email, password_hash, created_at, subscription_status
        FROM users
        WHERE id = ?
    ''', (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'email': row['email'],
            'password_hash': row['password_hash'],
            'created_at': row['created_at'],
            'subscription_status': row['subscription_status']
        }
    return None

def verify_password(user, password):
    """Verify user password"""
    return auth_verify_password(password, user['password_hash'])

def save_swing(user_id, video_filename, analysis_data):
    """Save a swing analysis to the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Convert analysis_data to JSON string
    analysis_json = json.dumps(analysis_data)
    
    cursor.execute('''
        INSERT INTO swings (user_id, video_filename, analysis_data)
        VALUES (?, ?, ?)
    ''', (user_id, video_filename, analysis_json))
    
    conn.commit()
    swing_id = cursor.lastrowid
    conn.close()
    
    return swing_id

def get_user_swings(user_id, limit=10):
    """Get user's swing history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, video_filename, analysis_data, created_at
        FROM swings
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (user_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    swings = []
    for row in rows:
        swings.append({
            'id': row['id'],
            'video_filename': row['video_filename'],
            'analysis_data': json.loads(row['analysis_data']),
            'created_at': row['created_at']
        })
    
    return swings

def get_swing_by_id(swing_id, user_id=None):
    """Get a specific swing by ID, optionally verify user ownership"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute('''
            SELECT id, user_id, video_filename, analysis_data, created_at
            FROM swings
            WHERE id = ? AND user_id = ?
        ''', (swing_id, user_id))
    else:
        cursor.execute('''
            SELECT id, user_id, video_filename, analysis_data, created_at
            FROM swings
            WHERE id = ?
        ''', (swing_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row['id'],
            'user_id': row['user_id'],
            'video_filename': row['video_filename'],
            'analysis_data': json.loads(row['analysis_data']),
            'created_at': row['created_at']
        }
    return None

