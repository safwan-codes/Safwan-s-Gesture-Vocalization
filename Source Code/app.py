from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from contextlib import closing
import subprocess
import sqlite3
import sys
import os
import db

# Always bind to localhost only — prevents /api/session-result from being
# reachable from the LAN while auth is not yet implemented (gap #7)
_HOST = '127.0.0.1'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SCRIPT_DIR, 'ASLModel.h5')
_USER_DB_PATH = os.path.join(_SCRIPT_DIR, 'users.db')
_SECRET_PATH = os.path.join(_SCRIPT_DIR, '.flask_secret')
_LABELS = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

dashboard_process = None
current_session_id = None   # tracks the active DB row


def _load_secret_key() -> bytes:
    """Persist the secret key across restarts so sessions survive a reload.
    (os.urandom on every boot logged everyone out and broke the debug reloader.)"""
    try:
        with open(_SECRET_PATH, 'rb') as f:
            key = f.read()
        if len(key) >= 24:
            return key
    except OSError:
        pass
    key = os.urandom(32)
    with open(_SECRET_PATH, 'wb') as f:
        f.write(key)
    try:
        os.chmod(_SECRET_PATH, 0o600)
    except OSError:
        pass
    return key


app = Flask(__name__)
app.secret_key = _load_secret_key()
db.init_db()   # creates sessions.db on first run, no-op thereafter


# ── User DB helpers ───────────────────────────────────────────────────────────

def _user_db():
    """Fresh connection per call with a busy timeout so concurrent requests
    wait instead of raising 'database is locked'."""
    conn = sqlite3.connect(_USER_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def init_user_db():
    with closing(_user_db()) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()


init_user_db()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if 'username' not in session:
            # fetch()/API callers need JSON, not a redirect to an HTML page —
            # otherwise response.json() dies with "Unexpected token '<'"
            if request.path.startswith('/api/') or request.path == '/start_session':
                return jsonify({'error': 'Not logged in'}), 401
            return redirect(url_for('index'))
        return view(*args, **kwargs)
    return wrapped


# ── Auth pages ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('register.html')


@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email', '').strip()
    password = request.form.get('pswd', '').strip()

    if not email or not password:
        return render_template('register.html',
                               login_error='Please enter your email and password.')

    with closing(_user_db()) as conn:
        row = conn.execute(
            "SELECT name, password FROM users WHERE email=?", (email,)
        ).fetchone()

    if row and check_password_hash(row[1], password):
        session['username'] = row[0]
        session['is_new_user'] = False
        return redirect(url_for('dashboard'))
    return render_template('register.html',
                           login_error='Invalid credentials or account does not exist.')


@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('txt', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('pswd', '').strip()

    if not email or not password or not name:
        return render_template('register.html', login_error='All fields are required.')

    hashed_password = generate_password_hash(password)

    try:
        with closing(_user_db()) as conn:
            conn.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (name, email, hashed_password)
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return render_template('register.html',
                               login_error='Email address is already registered.')

    session['username'] = name
    session['is_new_user'] = True
    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    stats = db.get_stats()

    sg_dir = os.path.join(_SCRIPT_DIR, 'SampleGestures')
    custom_gestures = len([
        f for f in os.listdir(sg_dir)
        if f.endswith('.png') and f != '..png'
    ]) if os.path.exists(sg_dir) else 0

    greeting = "Welcome" if session.get('is_new_user') else "Welcome back"

    return render_template('dashboard.html',
                           username=session.get('username', 'User'),
                           greeting=greeting,
                           total_sessions=stats['total_sessions'],
                           total_gestures=stats['total_gestures'],
                           total_sentences=stats['saved_count'],
                           custom_gestures=custom_gestures)


@app.route('/history')
@login_required
def history():
    return render_template('history.html', username=session.get('username', 'User'))


@app.route('/learn')
@login_required
def learn():
    return render_template('learn.html', current_user=session['username'])


@app.route('/tutorial')
@login_required
def tutorial():
    return render_template('tutorial.html', current_user=session['username'])


# ── ML Session lifecycle ──────────────────────────────────────────────────────

def _spawn_dashboard(mode: str, session_id: int) -> subprocess.Popen:
    dashboard_script = os.path.join(_SCRIPT_DIR, 'Dashboard.py')
    # sys.executable guarantees the same interpreter/venv that runs Flask —
    # a bare 'python' is not on PATH on many systems (notably macOS).
    return subprocess.Popen(
        [sys.executable, dashboard_script,
         '--session-id', str(session_id), '--mode', mode],
        cwd=_SCRIPT_DIR
    )


@app.route('/start_session', methods=['POST'])
@login_required
def start_session():
    global dashboard_process, current_session_id

    # Single-instance guard — MUST come before log_session_start
    # so a rejected "already running" click doesn't create a phantom DB row
    if dashboard_process is not None and dashboard_process.poll() is None:
        return jsonify({'status': 'error', 'message': 'Session already running.'})

    current_session_id = db.log_session_start('scanSent')   # default screen
    dashboard_process = _spawn_dashboard('scanSent', current_session_id)
    return jsonify({'status': 'success', 'session_id': current_session_id})


@app.route('/api/launch/<mode>', methods=['POST'])
@login_required
def launch_mode(mode):
    global dashboard_process, current_session_id

    if mode not in ('scanSingle', 'scanSent', 'createGest'):
        return jsonify({'error': f'Unknown mode: {mode}'}), 400

    # Kill any running process so we can switch modes
    if dashboard_process is not None and dashboard_process.poll() is None:
        try:
            dashboard_process.terminate()
            dashboard_process.wait(timeout=3)
        except Exception:
            try:
                dashboard_process.kill()
            except Exception:
                pass

    current_session_id = db.log_session_start(mode)
    dashboard_process = _spawn_dashboard(mode, current_session_id)
    return jsonify({'status': 'launched', 'mode': mode})


@app.route('/api/session-result', methods=['POST'])
def session_result():
    """
    Bridge endpoint: Dashboard.py POSTs here when a recognition session ends.
    Accepts JSON: { session_id, letters, gesture_count }
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id')
    letters = data.get('letters', '')
    gesture_count = int(data.get('gesture_count', 0))

    if session_id is None:
        return jsonify({'status': 'error', 'message': 'Missing session_id'}), 400

    db.log_session_end(int(session_id), letters, gesture_count)
    return jsonify({'status': 'ok'})


# ── Data API ──────────────────────────────────────────────────────────────────

@app.route('/api/stats')
@login_required
def api_stats():
    """Real KPI numbers from SQLite — no fabricated values."""
    return jsonify(db.get_stats())


@app.route('/api/recent-sessions')
@login_required
def api_recent_sessions():
    """Last 10 completed sessions for history table and chart."""
    return jsonify(db.get_recent_sessions(10))


@app.route('/api/sessions')
@login_required
def api_sessions():
    """Full session history for the History page."""
    return jsonify(db.get_recent_sessions(200))


@app.route('/api/model-status')
@login_required
def api_model_status():
    """Reports real model file metadata — no TF load required."""
    return jsonify(db.get_model_status(_MODEL_PATH, _LABELS))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # host=_HOST enforces localhost-only binding (gap #7).
    # Debug off by default — the Werkzeug debugger allows arbitrary code
    # execution and its held tracebacks can pin SQLite locks open.
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host=_HOST, port=5000)
