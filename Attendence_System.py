import tkinter as tk
from tkinter import messagebox
import mysql.connector
import face_recognition
import cv2
import numpy as np
import random
import time
import dlib
from scipy.spatial import distance
from datetime import datetime
import subprocess
import re

# Function to connect to the MySQL database
def connect_to_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="2004",
            database="attendance_system"
        )
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error connecting to database: {err}")
        raise

# Function to get the MAC address of the router using its IP address
def get_wifi_mac(ip):
    try:
        arp_output = subprocess.check_output(f"arp -a {ip}", shell=True, text=True)
        mac_pattern = re.compile(r"(([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2})")
        mac_match = mac_pattern.search(arp_output)
        if mac_match:
            return mac_match.group(0)
        else:
            return "MAC address not found."
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Merged function for liveness detection and face recognition
def liveness_and_recognition(employee_id):
    EAR_THRESHOLD = 0.3
    EAR_CONSEC_FRAMES = 3
    PROMPT_DISPLAY_TIME = 1  # Duration for which the blink prompt is shown (in seconds)
    PROMPT_INTERVAL = 5     # Interval between prompts (in seconds)
    REACTION_TIME = 5       # Time allowed for user reaction (in seconds)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    blink_count = 0
    liveness_passed = False
    blink_prompt = False
    prompt_start_time = None
    last_prompt_time = time.time()

    try:
        db = connect_to_db()
        cursor = db.cursor()

        cursor.execute("SELECT photo FROM company WHERE employee_id = %s", (employee_id,))
        result = cursor.fetchone()

        if result is None:
            messagebox.showerror("Error", "No photo found for the employee")
            return

        stored_photo_data = np.frombuffer(result[0], np.uint8)
        stored_photo_image = cv2.imdecode(stored_photo_data, cv2.IMREAD_COLOR)
        known_encoding = face_recognition.face_encodings(stored_photo_image)[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            for face in faces:
                shape = predictor(gray, face)
                shape = np.array([[p.x, p.y] for p in shape.parts()])

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                current_time = time.time()

                # Show the blink prompt at regular intervals
                if current_time - last_prompt_time >= PROMPT_INTERVAL:
                    last_prompt_time = current_time
                    prompt_start_time = current_time
                    blink_prompt = True
                    cv2.putText(frame, "Blink Now!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if prompt_start_time and current_time - prompt_start_time <= PROMPT_DISPLAY_TIME:
                    if ear < EAR_THRESHOLD:
                        blink_count += 1
                    else:
                        if blink_count >= EAR_CONSEC_FRAMES:
                            liveness_passed = True
                            blink_prompt = False
                            cv2.putText(frame, "Blink Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        blink_count = 0
                else:
                    if blink_prompt:
                        blink_prompt = False
                        cv2.putText(frame, "Prompt Over", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if liveness_passed:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    for face_encoding in face_encodings:
                        results = face_recognition.compare_faces([known_encoding], face_encoding)

                        if results[0]:
                            cap.release()
                            cv2.destroyAllWindows()
                            check_location(employee_id)
                            return

                for (x, y) in np.concatenate((leftEye, rightEye), axis=0):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            cv2.imshow("Liveness and Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if not liveness_passed:
            messagebox.showerror("Error", "Liveness check failed")
        else:
            messagebox.showerror("Error", "Face doesn't match")

    except Exception as e:
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showerror("Error", f"Error: {e}")

# Function to store attendance in table3
def store_attendance(employee_id, location):
    try:
        db = connect_to_db()
        cursor = db.cursor()

        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        cursor.execute("INSERT INTO table3 (employee_id, location, date, time) VALUES (%s, %s, %s, %s)",
                       (employee_id, location, date_str, time_str))
        db.commit()
        messagebox.showinfo("Success", "Attendance recorded successfully")
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error storing attendance: {err}")

# Function to check the location based on the MAC address and IP address
def check_location(employee_id):
    try:
        db = connect_to_db()
        cursor = db.cursor()

        cursor.execute("SELECT location FROM company WHERE employee_id = %s", (employee_id,))
        result = cursor.fetchone()

        if result is None:
            messagebox.showerror("Error", "Employee ID not found")
            return

        employee_location = result[0]

        cursor.execute("SELECT mac_address, ip_address FROM table2 WHERE location = %s", (employee_location,))
        result = cursor.fetchone()

        if result is None:
            messagebox.showerror("Error", "Location not found")
            return

        mac_address, ip_address = result

        retrieved_mac_address = get_wifi_mac(ip_address)

        if retrieved_mac_address is None:
            messagebox.showerror("Error", "Could not retrieve MAC address")
            return

        if retrieved_mac_address != mac_address:
            messagebox.showerror("Error", "You are not in your registered office")
            return

        store_attendance(employee_id, employee_location)
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error checking location: {err}")

# Function to handle the login process
def login(employee_id, password):
    try:
        db = connect_to_db()
        cursor = db.cursor()
        cursor.execute("SELECT password FROM company WHERE employee_id = %s", (employee_id,))
        result = cursor.fetchone()

        if result is None:
            messagebox.showerror("Error", "Employee ID not found")
            return

        stored_password = result[0]

        if password == stored_password:
            liveness_and_recognition(employee_id)
        else:
            messagebox.showerror("Error", "Password is incorrect")
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error during login: {err}")

# GUI setup
def setup_gui():
    root = tk.Tk()
    root.title("Attendance System Login")

    tk.Label(root, text="Employee ID").grid(row=0, column=0)
    employee_id_entry = tk.Entry(root)
    employee_id_entry.grid(row=0, column=1)

    tk.Label(root, text="Password").grid(row=1, column=0)
    password_entry = tk.Entry(root, show='*')
    password_entry.grid(row=1, column=1)

    def handle_login():
        employee_id = employee_id_entry.get()
        password = password_entry.get()
        login(employee_id, password)

    tk.Button(root, text="Login", command=handle_login).grid(row=2, columnspan=2)

    root.mainloop()

setup_gui()
