import cv2
import face_recognition
import numpy as np
import pickle
import os
import csv
from datetime import datetime, time
import json
import time as py_time # For sleep
from flask import Flask, Response, render_template_string, jsonify

# --- Configuration (Must match paths used for encoding) ---
ESP32_STREAM_URL = "http://192.168.137.161:81/stream" # ⚠️ UPDATE THIS with your ESP32's IP
ENCODINGS_PATH = "encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"
CONFIG_FILE = "config.json"
# ---------------------
# --- Global State & Initialization ---
app = Flask(__name__)
known_face_encodings = []
known_face_names = []
names_marked_today = set()
ATTENDANCE_START_TIME = time(8, 0)
ATTENDANCE_END_TIME = time(11, 0)
last_check_date = datetime.now().strftime("%Y-%m-%d")

# --- Helper Functions ---

def load_config():
    """Loads the time window from config.json and updates global times."""
    global ATTENDANCE_START_TIME, ATTENDANCE_END_TIME
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            start = datetime.strptime(config['start_time'], "%H:%M").time()
            end = datetime.strptime(config['end_time'], "%H:%M").time()
            ATTENDANCE_START_TIME = start
            ATTENDANCE_END_TIME = end
            print(f"[INFO] Attendance Window Loaded: {start.strftime('%H:%M')} - {end.strftime('%H:%M')}")
    except Exception as e:
        print(f"[ERROR] Could not load config.json: {e}")
        print("[INFO] Using default times 08:00 - 11:00")

def load_encodings():
    """Loads known faces into global variables."""
    global known_face_encodings, known_face_names
    print("[INFO] Loading encodings...")
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
    except FileNotFoundError:
        print(
            f"[ERROR] Encodings file not found at {ENCODINGS_PATH}. Please run encode_faces.py"
        )
        # Continue with empty lists
        known_face_encodings = []
        known_face_names = []


def get_current_date():
    """Returns the current date as a string."""
    return datetime.now().strftime("%Y-%m-%d")


def reset_daily_log():
    """Resets the set of marked names at the start of a new day and initializes attendance file."""
    global names_marked_today, last_check_date
    current_date = get_current_date()
    header = ["Name", "Date", "Time", "Status"]

    # Ensure the attendance file exists and has a header
    if not os.path.exists(ATTENDANCE_FILE):
        print(f"[INFO] No attendance file found. Creating {ATTENDANCE_FILE}")
        with open(ATTENDANCE_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    current_day_entries = []
    try:
        # Read the file to repopulate the set if it's the same day
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            
            for row in lines[1:]: # Skip header
                if len(row) > 1 and row[1] == current_date:
                    current_day_entries.append(row[0])

    except Exception as e:
        print(f"[WARNING] Error reading {ATTENDANCE_FILE} for daily reset: {e}")

    if last_check_date != current_date:
        print(f"[INFO] New day ({current_date}). Resetting daily attendance log.")
        names_marked_today.clear()
        last_check_date = current_date # Update the tracker
    
    if current_day_entries:
        # If it's the same day, update the set based on what was read
        names_marked_today.update(current_day_entries)
        print(f"[INFO] Resuming attendance for {current_date}. Already marked: {names_marked_today}")
    elif last_check_date == current_date:
        # Same day, but nothing read (maybe only header)
        names_marked_today.clear()
        print(f"[INFO] Resuming attendance for {current_date}. No one marked yet.")

    last_check_date = current_date


def log_attendance(name, status):
    """Logs the attendance to the CSV file."""
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M:%S")

    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, current_date, current_time_str, status])


def check_and_log_attendance(name):
    """Checks the time window and logs attendance if not already marked."""
    global names_marked_today

    if name not in names_marked_today:
        current_time = datetime.now().time()
        status = ""

        if current_time < ATTENDANCE_START_TIME:
            status = "Early"
        elif current_time > ATTENDANCE_END_TIME:
            status = "Absent"
        else:
            status = "Present"

        print(
            f"[INFO] Marking attendance for {name} at {current_time.strftime('%H:%M:%S')} with status: {status}"
        )
        log_attendance(name, status)
        names_marked_today.add(name)

# --- Video Streaming Generator ---

def generate_frames():
    """Generates the motion JPEG stream for the web interface."""
    global last_check_date
    video_capture = cv2.VideoCapture(ESP32_STREAM_URL)

    if not video_capture.isOpened():
        print(f"[FATAL] Could not open video stream at {ESP32_STREAM_URL}. Trying local camera...")
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
             print("[FATAL] Local camera also failed. The system cannot run.")
             return 
        else:
             print("[INFO] Using local camera (index 0) as fallback.")


    print("[INFO] Video stream connected. Streaming...")
    process_this_frame = True

    while True:
        # --- Handle Midnight Rollover ---
        current_date = get_current_date()
        if current_date != last_check_date:
            reset_daily_log()
        # -------------------------------

        success, frame = video_capture.read()
        
        # --- ROBUST FRAME/CONNECTION CHECK ---
        if not success or frame is None or frame.size == 0:
            print("[WARNING] Lost connection or empty frame. Retrying in 2s...")
            video_capture.release()
            py_time.sleep(2) 
            video_capture = cv2.VideoCapture(ESP32_STREAM_URL)
            if not video_capture.isOpened():
                print("[ERROR] Reconnection failed. Keeping connection attempt open.")
            continue
        # -------------------------------------

        if process_this_frame:
            
            # Ensure frame is 8-bit BGR
            if frame.dtype != np.uint8:
                try:
                    frame = frame.astype(np.uint8)
                except Exception as e:
                    print(f"[ERROR] Could not force frame to uint8 data type: {e}. Skipping frame.")
                    process_this_frame = not process_this_frame
                    continue

            # Frame Integrity Check & Conversion
            if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"[WARNING] Malformed frame (shape: {frame.shape}). Skipping processing.")
                process_this_frame = not process_this_frame
                continue
            
            # Resize frame for faster processing (1/4 size)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert BGR to RGB (face_recognition requirement)
            try:
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            except cv2.error as e:
                print(f"[ERROR] OpenCV color conversion failed ({e}). Skipping frame.")
                process_this_frame = not process_this_frame 
                continue

            # Find faces and encodings
            face_locations = []
            face_encodings = []
            face_names = []
            
            if known_face_encodings:
                try:
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(
                        rgb_small_frame, face_locations
                    )
                except RuntimeError as e:
                    print(f"[ERROR] Face Recognition Runtime Error: {e}. Skipping frame.")
                    pass

            for face_encoding in face_encodings:
                name = "Unknown"
                
                # Match faces
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                
                if face_recognition.compare_faces(known_face_encodings, face_encoding)[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    check_and_log_attendance(name)

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results on the full-size frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" and name not in names_marked_today else (0, 0, 255)
            if name in names_marked_today:
                color = (255, 255, 0) # Yellow for already marked

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            status_text = name
            if name != "Unknown" and name in names_marked_today:
                status_text += " (MARKED)"
            
            cv2.putText(
                frame, status_text, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1
            )

        (flag, encodedImage) = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

# --- New Status Data Route for AJAX ---

@app.route("/status_data")
def get_status_data():
    """Provides real-time attendance statistics as JSON for the UI."""
    log_entries = []
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            # Find all entries for today and keep the last 10
            today_date = get_current_date()
            recent_logs = [row for row in lines[1:] if len(row) > 1 and row[1] == today_date]
            
            # Get only the last 10 entries and reverse them for newest-first display
            for row in recent_logs[-10:][::-1]: 
                 # Expecting: [Name, Date, Time, Status]
                if len(row) == 4:
                    log_entries.append({'name': row[0], 'time': row[2], 'status': row[3]})
    except Exception as e:
        print(f"[WARNING] Error reading CSV for status data: {e}")
        pass

    return jsonify({
        'current_date': get_current_date(),
        'current_time': datetime.now().strftime('%H:%M:%S'),
        'start_time': ATTENDANCE_START_TIME.strftime('%H:%M'),
        'end_time': ATTENDANCE_END_TIME.strftime('%H:%M'),
        'marked_count': len(names_marked_today),
        'total_known': len(known_face_names),
        'recent_logs': log_entries
    })


# --- Flask Routes (Enhanced UI) ---

@app.route("/")
def index():
    """Renders the single HTML page with the enhanced interface."""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GEC Talkal - Face Attendance</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
        html, body { 
            height: 100%; 
            margin: 0; 
            font-family: 'Inter', sans-serif;
        }
        /* Custom Keyframe for the pulsing border effect */
        @keyframes pulse-glow {
            0% { box-shadow: 0 0 10px rgba(66, 153, 225, 0.5); }
            50% { box-shadow: 0 0 20px rgba(99, 102, 241, 0.9), 0 0 40px rgba(99, 102, 241, 0.4); }
            100% { box-shadow: 0 0 10px rgba(66, 153, 225, 0.5); }
        }
        .live-stream-glow {
            animation: pulse-glow 3s infinite ease-in-out;
            transition: box-shadow 0.5s;
        }
        /* Custom Scrollbar for Log */
        .log-scroll::-webkit-scrollbar {
            width: 6px;
        }
        .log-scroll::-webkit-scrollbar-thumb {
            background-color: #4f46e5; /* indigo-600 */
            border-radius: 3px;
        }
        .log-scroll::-webkit-scrollbar-track {
            background-color: #374151; /* gray-700 */
        }
        .text-shadow {
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 flex flex-col items-center p-4 sm:p-8 min-h-screen">

    <!-- Header & Branding -->
    <header class="text-center mb-8 w-full max-w-4xl">
        <h1 class="text-3xl sm:text-4xl font-extrabold text-white-500 tracking-wider transition duration-5000">
            Government Engineering College, Talkal
        </h1>
        <h2 class="text-xl sm:text-2xl font-bold text-indigo mt-1 transition duration-500 ">
            Computer Science And Engineering
        </h2>
        <!-- MODIFIED: Changed text-indigo-400 to text-red-500 and removed bg-gray-800 -->
        <h3 class="text-3xl sm:text-3xl font-extrabold text-Aquamarine-500 mt-2 p-1 rounded-lg inline-block shadow-lg">
            FACE RECOGNITION ATTENDANCE SYSTEM
        </h3>
    </header>

    <!-- Main Content Grid -->
    <main class="w-full max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <!-- Live Video Stream (2/3 width on large screens) -->
        <section class="lg:col-span-2 bg-gray-800 p-2 rounded-2xl shadow-2xl live-stream-glow transition duration-500">
            <h4 class="text-lg font-medium text-center mb-2 text-indigo-300">LIVE FEED & DETECTION</h4>
            <div class="aspect-video bg-black overflow-hidden rounded-xl">
                <!-- The video feed is served by the /video_feed route -->
                <img id="videoFeed" src="{{ url_for('video_feed') }}" alt="Live Camera Feed" class="w-full h-full object-cover">
            </div>
            <p class="text-xs text-center text-gray-500 mt-2">Source: {{ esp32_url }}</p>
        </section>

        <!-- Status Panel (1/3 width on large screens) -->
        <aside class="lg:col-span-1 space-y-6">
            
            <!-- Real-Time Clock & Date -->
            <div class="bg-indigo-600 p-4 rounded-xl shadow-xl text-center">
                <p id="currentDate" class="text-sm font-light opacity-80"></p>
                <p id="currentTime" class="text-5xl font-extrabold text-white text-shadow transition duration-500"></p>
                <p class="text-sm font-medium mt-1">System Time</p>
            </div>

            <!-- Attendance Summary Card -->
            <div class="bg-gray-800 p-5 rounded-xl shadow-xl border border-indigo-500/50">
                <h4 class="text-xl font-semibold text-white mb-3 flex items-center">
                    Attendance Summary
                </h4>
                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-gray-700 p-3 rounded-lg text-center shadow-inner">
                        <p id="markedCount" class="text-3xl font-bold text-green-400">0</p>
                        <p class="text-xs text-gray-400">Marked Today</p>
                    </div>
                    <div class="bg-gray-700 p-3 rounded-lg text-center shadow-inner">
                        <p id="totalKnown" class="text-3xl font-bold text-yellow-400">0</p>
                        <p class="text-xs text-gray-400">Known Faces</p>
                    </div>
                </div>
                <div class="mt-4 text-sm text-gray-300">
                    <p>Window: <span id="timeWindow" class="font-bold text-indigo-300">{{ start_time }} - {{ end_time }}</span></p>
                    <p>Log File: <code class="text-xs text-pink-400">{{ attendance_file }}</code></p>
                </div>
            </div>

            <!-- Real-Time Activity Log -->
            <div class="bg-gray-800 p-5 rounded-xl shadow-xl border border-gray-700">
                <h4 class="text-xl font-semibold text-white mb-3">Recent Activity</h4>
                <div id="activityLog" class="h-48 overflow-y-auto log-scroll space-y-3">
                    <!-- Log entries populated by JavaScript -->
                    <p class="text-sm text-gray-500 text-center pt-8">Loading real-time log...</p>
                </div>
            </div>
        </aside>
    </main>
    
    <!-- JavaScript for Real-Time Updates -->
    <script>
        // Function to update the UI with fresh status data
        async function updateStatus() {
            try {
                const response = await fetch('{{ url_for("get_status_data") }}');
                const data = await response.json();

                // 1. Update Clock and Date
                document.getElementById('currentTime').textContent = data.current_time;
                document.getElementById('currentDate').textContent = data.current_date;

                // 2. Update Summary
                document.getElementById('markedCount').textContent = data.marked_count;
                document.getElementById('totalKnown').textContent = data.total_known;

                // 3. Update Activity Log
                const logContainer = document.getElementById('activityLog');
                logContainer.innerHTML = ''; // Clear existing log

                if (data.recent_logs.length === 0) {
                    logContainer.innerHTML = '<p class="text-sm text-gray-500 text-center pt-8">No attendance entries for today yet.</p>';
                    return;
                }

                data.recent_logs.forEach(log => {
                    let colorClass = 'text-green-400';
                    if (log.status.includes('Late')) {
                        colorClass = 'text-red-400';
                    } else if (log.status.includes('Early')) {
                        colorClass = 'text-yellow-400';
                    } else if (log.status.includes('Unknown')) {
                        colorClass = 'text-gray-400';
                    }
                    
                    const logItem = document.createElement('div');
                    logItem.className = 'flex justify-between items-center p-2 bg-gray-700 rounded-md shadow-md hover:bg-gray-600 transition duration-300';
                    logItem.innerHTML = `
                        <p class="text-sm font-semibold text-white">${log.name}</p>
                        <div class="text-right">
                            <span class="text-xs font-mono text-gray-400">${log.time}</span>
                            <span class="ml-2 text-xs font-bold ${colorClass}">${log.status.toUpperCase()}</span>
                        </div>
                    `;
                    logContainer.appendChild(logItem);
                });

            } catch (error) {
                console.error("Error fetching status data:", error);
                document.getElementById('activityLog').innerHTML = '<p class="text-sm text-red-500 text-center pt-8">Error loading log data.</p>';
            }
        }

        // Run the update function immediately and then every 2 seconds
        updateStatus();
        setInterval(updateStatus, 2000); 
    </script>
</body>
</html>
    """, esp32_url=ESP32_STREAM_URL, 
       attendance_file=ATTENDANCE_FILE,
       start_time=ATTENDANCE_START_TIME.strftime('%H:%M'),
       end_time=ATTENDANCE_END_TIME.strftime('%H:%M'))

@app.route("/video_feed")
def video_feed():
    """Video streaming route. It connects the generator function to the HTTP response."""
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Main execution ---
if __name__ == '__main__':
    # Initialize global state before starting the Flask server
    load_config()
    load_encodings()
    reset_daily_log()

    print("\n[FLASK] Starting server...")
    print(f"[FLASK] Access the attendance UI at: http://127.0.0.1:5000/")
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
