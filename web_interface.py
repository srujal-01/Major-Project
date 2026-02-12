from flask import Flask, render_template, jsonify
import csv
import os
import json

app = Flask(__name__)

ATTENDANCE_FILE = "attendance.csv"
CONFIG_FILE = "config.json"

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/get_attendance')
def get_attendance():
    """API endpoint to get attendance data from CSV."""
    data = []
    try:
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            try:
                next(reader) # Skip header
            except StopIteration:
                return jsonify([]) # File is empty
            
            # Read rows in reverse to show newest first
            for row in reversed(list(reader)):
                data.append(row)
    except FileNotFoundError:
        print(f"[WARN] {ATTENDANCE_FILE} not found. Returning empty list.")
        return jsonify([]) # File doesn't exist yet
    except Exception as e:
        print(f"[ERROR] Error reading attendance file: {e}")
        return jsonify({"error": str(e)}), 500
        
    return jsonify(data)

@app.route('/get_time_window')
def get_time_window():
    """API endpoint to get the current time window from config.json."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            response = {
                "start": config.get('start_time', '00:00'),
                "end": config.get('end_time', '00:00')
            }
        return jsonify(response)
    except FileNotFoundError:
        print(f"[ERROR] {CONFIG_FILE} not found.")
        return jsonify({"error": f"{CONFIG_FILE} not found."}), 404
    except Exception as e:
        print(f"[ERROR] Error reading config file: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Running on port 5001 so it doesn't conflict
    # with any other service.
    print("[INFO] Starting Admin Dashboard Server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
