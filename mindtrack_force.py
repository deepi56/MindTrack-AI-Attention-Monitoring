import cv2
import mediapipe as mp
import numpy as np
import csv
import threading
import winsound
import time
from datetime import datetime
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
EAR_THRESHOLD = 0.25
CLOSED_FRAME_LIMIT = 5
YAW_THRESHOLD = 20
EAR_HISTORY_SIZE = 5

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

CSV_FILE = "attention_log.csv"

# ================= ATTENTION TRACKER =================
class AttentionTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.draw = mp.solutions.drawing_utils
        self.spec = self.draw.DrawingSpec(thickness=1, circle_radius=1)

        self.ear_history = []
        self.closed_frames = 0
        self.alarm_active = False

        # State tracking for clean CSV
        self.current_state = None
        self.state_start_time = None

        # Data for visualization
        self.states = []
        self.timestamps = []

        self._init_csv()

    # ================= CSV =================
    def _init_csv(self):
        self.csv_file = open(CSV_FILE, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["State", "Start Time", "End Time", "Duration (sec)"])

    # ================= ALARM =================
    def _alarm_loop(self):
        while self.alarm_active:
            winsound.Beep(1000, 700)

    def start_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            threading.Thread(target=self._alarm_loop, daemon=True).start()

    def stop_alarm(self):
        self.alarm_active = False

    # ================= CALCULATIONS =================
    def eye_aspect_ratio(self, landmarks, eye_points):
        p = [(landmarks[i].x, landmarks[i].y) for i in eye_points]
        v1 = np.linalg.norm(np.subtract(p[1], p[5]))
        v2 = np.linalg.norm(np.subtract(p[2], p[4]))
        h = np.linalg.norm(np.subtract(p[0], p[3]))
        return (v1 + v2) / (2.0 * h)

    def head_yaw(self, landmarks, w):
        nose_x = landmarks[1].x * w
        left_x = landmarks[33].x * w
        right_x = landmarks[263].x * w
        return nose_x - ((left_x + right_x) / 2)

    # ================= STATE LOGIC =================
    def determine_state(self, ear, yaw):
        if ear < EAR_THRESHOLD:
            self.closed_frames += 1
        else:
            self.closed_frames = 0

        if self.closed_frames >= CLOSED_FRAME_LIMIT:
            return "Sleepy"
        elif abs(yaw) > YAW_THRESHOLD:
            return "Distracted"
        else:
            return "Focused"

    # ================= FRAME PROCESS =================
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame

        for face in results.multi_face_landmarks:
            self.draw.draw_landmarks(
                frame,
                face,
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                self.spec,
                self.spec
            )

            landmarks = face.landmark

            left_ear = self.eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = self.eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2

            self.ear_history.append(avg_ear)
            if len(self.ear_history) > EAR_HISTORY_SIZE:
                self.ear_history.pop(0)

            smooth_ear = np.mean(self.ear_history)
            yaw = self.head_yaw(landmarks, w)

            state = self.determine_state(smooth_ear, yaw)

            # Alarm & color
            if state == "Focused":
                self.stop_alarm()
                color = (0, 255, 0)
            elif state == "Distracted":
                self.start_alarm()
                color = (0, 165, 255)
            else:
                self.start_alarm()
                color = (0, 0, 255)

            # ================= CLEAN CSV LOGGING =================
            current_time = time.time()

            if self.current_state is None:
                self.current_state = state
                self.state_start_time = current_time

            elif state != self.current_state:
                duration = current_time - self.state_start_time
                self.writer.writerow([
                    self.current_state,
                    datetime.fromtimestamp(self.state_start_time).strftime("%H:%M:%S"),
                    datetime.fromtimestamp(current_time).strftime("%H:%M:%S"),
                    round(duration, 1)
                ])

                self.current_state = state
                self.state_start_time = current_time

            # Store for visualization
            self.states.append(state)
            self.timestamps.append(current_time)

            cv2.putText(
                frame,
                f"State: {state}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        return frame

    # ================= FINAL REPORT =================
    def generate_report(self):
        durations = {"Focused": 0, "Distracted": 0, "Sleepy": 0}

        for i in range(1, len(self.states)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            durations[self.states[i - 1]] += dt

        labels = ["Focused 🙂", "Distracted 😐", "Sleepy 😴"]
        times = [
            durations["Focused"],
            durations["Distracted"],
            durations["Sleepy"]
        ]

        colors = ["#4CAF50", "#FFC107", "#F44336"]

        plt.figure(figsize=(7, 7))
        plt.pie(
            times,
            labels=labels,
            colors=colors,
            autopct=lambda p: f"{p:.0f}%",
            startangle=90,
            textprops={'fontsize': 12}
        )

        plt.title(
            "MindTrack – Attention Summary\n(Green = Good, Red = Not OK)",
            fontsize=14,
            fontweight="bold"
        )

        plt.savefig("attention_pie_report.png")
        plt.show()

        print("✅ Pie chart saved as attention_pie_report.png")

    # ================= CLEANUP =================
    def cleanup(self):
        if self.current_state is not None:
            end_time = time.time()
            duration = end_time - self.state_start_time
            self.writer.writerow([
                self.current_state,
                datetime.fromtimestamp(self.state_start_time).strftime("%H:%M:%S"),
                datetime.fromtimestamp(end_time).strftime("%H:%M:%S"),
                round(duration, 1)
            ])

        self.csv_file.close()
        self.stop_alarm()

# ================= MAIN =================
def main():
    tracker = AttentionTracker()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = tracker.process_frame(frame)

        cv2.imshow("MindTrack – Attention Detection", frame)

        if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.cleanup()
    tracker.generate_report()

if __name__ == "__main__":
    main()
