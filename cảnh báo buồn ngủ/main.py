import cv2
import dlib
import time
import os
import pygame
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from imutils import face_utils
from scipy.spatial import distance

# Ngưỡng
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 15
MAR_THRESHOLD = 0.6

ALARM_SOUND = "alarm.mp3"
COUNTER = 0
ALARM_ON = False
YAWN_ON = False

# Tính EAR (mắt)
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Tính MAR ( ngápngáp)
def calculate_MAR(mouth):
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    D = distance.euclidean(mouth[12], mouth[16])
    return (A + B + C) / (3.0 * D)

# Âm thanh cảnh báo
def sound_alarm():
    global ALARM_ON
    if not os.path.exists(ALARM_SOUND):
        print(f"[ERROR] File âm thanh '{ALARM_SOUND}' không tồn tại.")
        ALARM_ON = False
        return
    pygame.mixer.init()
    pygame.mixer.music.load(ALARM_SOUND)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    ALARM_ON = False

# Thiét lập giao diện  
class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Giám sát buồn ngủ & ngáp")
        self.video_frame = tk.Label(root)
        self.video_frame.pack()

        self.status_label = tk.Label(root, text="TRẠNG THÁI: Chờ bắt đầu", font=("Arial", 14), fg="blue")
        self.status_label.pack(pady=5)

        self.ear_label = tk.Label(root, text="EAR: --", font=("Arial", 12))
        self.ear_label.pack()

        self.mar_label = tk.Label(root, text="MAR: --", font=("Arial", 12))
        self.mar_label.pack()

        self.start_btn = tk.Button(root, text="Bắt đầu giám sát", command=self.start_monitor)
        self.start_btn.pack(side=tk.LEFT, padx=20, pady=10)

        self.quit_btn = tk.Button(root, text="Thoát", command=self.quit_app)
        self.quit_btn.pack(side=tk.RIGHT, padx=20, pady=10)

        self.cap = None
        self.running = False

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        self.mStart, self.mEnd = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    def start_monitor(self):
        if not self.running:
            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.update_frame()

    def update_frame(self):
        global COUNTER, ALARM_ON, YAWN_ON
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)

        status_text = "TRẠNG THÁI: Bình thường"
        color = "green"
        ear, mar = 0, 0

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            mouth = shape[self.mStart:self.mEnd]

            leftEAR = calculate_EAR(leftEye)
            rightEAR = calculate_EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = calculate_MAR(mouth)

            # Viền
            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EAR_CONSEC_FRAMES and not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=sound_alarm, daemon=True).start()
                    status_text = "CẢNH BÁO! BUỒN NGỦ!"
                    color = "red"
                    with open("log.txt", "a", encoding="utf-8") as log:
                        log.write(f"[{time.ctime()}] BUỒN NGỦ - EAR: {ear:.2f} MAR: {mar:.2f}\n")
            else:
                COUNTER = 0

            if mar > MAR_THRESHOLD and not YAWN_ON:
                YAWN_ON = True
                threading.Thread(target=sound_alarm, daemon=True).start()
                status_text = "CẢNH BÁO NGÁP!"
                color = "orange"
                with open("log.txt", "a", encoding="utf-8") as log:
                    log.write(f"[{time.ctime()}] NGÁP - EAR: {ear:.2f} MAR: {mar:.2f}\n")
            elif mar <= MAR_THRESHOLD:
                YAWN_ON = False

        # Cập nhật GUI
        self.status_label.config(text=status_text, fg=color)
        self.ear_label.config(text=f"EAR: {ear:.2f}")
        self.mar_label.config(text=f"MAR: {mar:.2f}")

        # Cập nhật ảnh
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_frame.imgtk = img
        self.video_frame.configure(image=img)

        if self.running:
            self.root.after(10, self.update_frame)

    def quit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

# Khởi động
if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.mainloop()
