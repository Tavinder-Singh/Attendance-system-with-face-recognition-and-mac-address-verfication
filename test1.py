import tkinter as tk
from tkinter import messagebox
import mysql.connector
import face_recognition
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from datetime import datetime
import subprocess
import re
import time

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
        return None

# Function to get the MAC address of the router using its IP address
def get_wifi_mac(ip):
    try:
        arp_output = subprocess.check_output(f"arp -a {ip}", shell=True, text=True)
        mac_pattern = re.compile(r"(([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2})")
        mac_match = mac_pattern.search(arp_output)
        return mac_match.group(0) if mac_match else None
    except subprocess.CalledProcessError:
        return None

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function for liveness detection and face recognition
def liveness_and_recognition(employee_id):
    EAR_THRESHOLD = 0.3
    EAR_CONSEC_FRAMES = 3
    PROMPT_DISPLAY_TIME = 1  # Blink prompt duration
    PROMPT_INTERVAL = 5  # Time between prompts
    REACTION_TIME = 5  # User response time

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    blink_count = 0
    liveness_passed = False
    blink_prompt = False
    last_prompt_time = time.time()

    db = connect_to_db()
    if db is None:
        return
    cursor = db.cursor()

    cursor.execute("SELECT photo FROM company WHERE employee_id = %s", (employee_id,))
    result = cursor.fetchone()
    cursor.close()
    db.close()

    if result is None:
        messagebox.showerror("Error", "No photo found for the employee")
        return

    stored_photo_data = np.frombuffer(result[0], np.uint8)
    stored_photo_image = cv2.imdecode(stored_photo_data, cv2.IMREAD_COLOR)

    known_encodings = face_recognition.face_encodings(stored_photo_image)
    if not known_encodings:
        messagebox.showerror("Error", "No face detected in stored image")
        return

    known_encoding = known_encodings[0]

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

            if current_time - last_prompt_time >= PROMPT_INTERVAL:
                last_prompt_time = current_time
                blink_prompt = True
                cv2.putText(frame, "Blink Now!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if blink_prompt and ear < EAR_THRESHOLD:
                blink_count += 1
                if blink_count >= EAR_CONSEC_FRAMES:
                    liveness_passed = True
                    blink_prompt = False
                    cv2.putText(frame, "Blink Detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if liveness_passed:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_frame)

                if face_encodings:
                    match = face_recognition.compare_faces([known_encoding], face_encodings[0])[0]
                    if match:
                        cap.release()
                        cv2.destroyAllWindows()
                        check_location(employee_id)
                        return

        cv2.imshow("Liveness and Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showerror("Error", "Face doesn't match or liveness failed")

# Function to store attendance
def store_attendance(employee_id, location):
    db = connect_to_db()
    if db is None:
        return
    cursor = db.cursor()

    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')

    cursor.execute("INSERT INTO table3 (employee_id, location, date, time) VALUES (%s, %s, %s, %s)",
                   (employee_id, location, date_str, time_str))
    db.commit()
    cursor.close()
    db.close()
    messagebox.showinfo("Success", "Attendance recorded successfully")

# Function to check the location
def check_location(employee_id):
    db = connect_to_db()
    if db is None:
        return
    cursor = db.cursor()

    cursor.execute("SELECT location FROM company WHERE employee_id = %s", (employee_id,))
    result = cursor.fetchone()

    if result:
        employee_location = result[0]
        cursor.execute("SELECT mac_address, ip_address FROM table2 WHERE location = %s", (employee_location,))
        result = cursor.fetchone()

        if result:
            mac_address, ip_address = result
            retrieved_mac_address = get_wifi_mac(ip_address)

            if retrieved_mac_address and retrieved_mac_address == mac_address:
                store_attendance(employee_id, employee_location)
            else:
                messagebox.showerror("Error", "Not in registered office")

    cursor.close()
    db.close()

# Login function
def login(employee_id, password):
    db = connect_to_db()
    if db is None:
        return
    cursor = db.cursor()

    cursor.execute("SELECT password FROM company WHERE employee_id = %s", (employee_id,))
    result = cursor.fetchone()

    if result and password == result[0]:
        liveness_and_recognition(employee_id)
    else:
        messagebox.showerror("Error", "Incorrect ID or password")

    cursor.close()
    db.close()

# GUI setup
def setup_gui():
    root = tk.Tk()
    root.title("Attendance System")

    tk.Label(root, text="Employee ID").grid(row=0, column=0)
    employee_id_entry = tk.Entry(root)
    employee_id_entry.grid(row=0, column=1)

    tk.Label(root, text="Password").grid(row=1, column=0)
    password_entry = tk.Entry(root, show='*')
    password_entry.grid(row=1, column=1)

    tk.Button(root, text="Login", command=lambda: login(employee_id_entry.get(), password_entry.get())).grid(row=2, columnspan=2)

    root.mainloop()

setup_gui()
