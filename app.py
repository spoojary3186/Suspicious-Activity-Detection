
import os
import sys
import sqlite3
import threading
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "user_data.db"
UPLOAD_DIR = BASE_DIR / "test"
DETECT_SCRIPT = BASE_DIR / "detect.py"
PY = sys.executable  # current python

# Ensure upload folder exists
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)


# ==========================
# DATABASE
# ==========================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user(
            name TEXT PRIMARY KEY,
            password TEXT,
            mobile TEXT,
            email TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ==========================
# HELPERS
# ==========================
def run_detect_on_source(source: str):
    """
    Run detect.py on a given source.
    - For live: source = "0"
    - For analyse video: source = full path to saved video
    """
    import subprocess

    cmd = [
        PY,
        str(DETECT_SCRIPT),
        "--source", str(source),
        "--device", "0",
        "--half",
        "--view-img"   # IMPORTANT: opens OpenCV window
    ]
    print("Running detect.py:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("detect.py exited with error:", e)
    except Exception as e:
        print("Unexpected error running detect.py:", e)


# ==========================
# ROUTES
# ==========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/userlog", methods=["POST"])
def userlog():
    name = request.form.get("name", "").strip()
    password = request.form.get("password", "").strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user WHERE name=? AND password=?", (name, password))
    result = cursor.fetchone()
    conn.close()

    if result is None:
        return render_template("index.html", msg="Incorrect Username or Password")

    # Login success → show Analyse + Live buttons
    return render_template("userlog.html")


@app.route("/userreg", methods=["POST"])
def userreg():
    name = request.form.get("name", "").strip()
    password = request.form.get("password", "").strip()
    mobile = request.form.get("phone", "").strip()
    email = request.form.get("email", "").strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO user VALUES (?, ?, ?, ?)",
            (name, password, mobile, email)
        )
        conn.commit()
        msg = "Successfully Registered"
    except sqlite3.IntegrityError:
        msg = "Username already exists"

    conn.close()
    return render_template("index.html", msg=msg)


# ==========================
# ANALYSE (OFFLINE VIDEO)
# ==========================
import threading
import A_Recognition

@app.route('/video', methods=['POST'])
def video_route():
    if 'file' not in request.files:
        return render_template("userlog.html")  # stay on page

    file = request.files['file']
    if file.filename == '':
        return render_template("userlog.html")

    save_path = os.path.join("test", file.filename)
    file.save(save_path)
    print("Saved upload:", save_path)

    # Run action recognition in background so popup can appear
    threading.Thread(
        target=A_Recognition.analyse,
        args=(save_path,),
        daemon=True
    ).start()

    # Just reload the userlog.html page - no message
    return render_template(
        "userlog.html",
        msg="Analysing... Please wait.")




# ==========================
# LIVE STREAMING
# ==========================
@app.route("/Live")
def Live():
    """
    Start live webcam detection using detect.py --source 0
    """
    t = threading.Thread(
        target=run_detect_on_source,
        args=("0",),
        daemon=True
    )
    t.start()

    return render_template(
        "userlog.html",
        msg="Live Streaming Starting."
    )


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    print("Starting Flask app:", PY)
    app.run(debug=True, use_reloader=False)
