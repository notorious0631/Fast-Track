import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import pickle
import time
import logging
import serial

# Setup logging
logging.basicConfig(filename="video_feed.log", level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
dataset_path = "dataset"
trainer_path = "trainer"
attendance_file = "attendance.csv"
cascade_path = "haarcascade_frontalface_default.xml"  # Update with your path

# Ensure directories exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

# Initialize face cascade
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise Exception("Error loading Haar cascade file")

# GUI Class
class AttendanceSystem:
    def __init__(self, root):  # <== use __init__, not _init_
        self.root = root
        self.root.title("Face Detection Attendance System")
        self.root.geometry("800x600")
# Arduino connection
        try:
            self.arduino = serial.Serial('COM5', 9600, timeout=1)  
            time.sleep(2)  # Wait for Arduino to initialize
        except:
            self.arduino = None
            logging.warning("Could not connect to Arduino")


        # User ID and Name
        self.user_id = 1
        if os.path.exists("user_id.pkl"):
            with open("user_id.pkl", "rb") as f:
                self.user_id = pickle.load(f)

        # Camera control
        self.cap = None
        self.camera_running = False
        self.recognizer = None
        self.names = {}
        self.load_names()

        # Add user capture state
        self.capturing = False
        self.capture_count = 0
        self.capture_user_id = None
        self.capture_name = None
        self.capture_cancelled = False

        # New user lock for 30 seconds
        self.new_user_lock = {}  # {name: registration_timestamp}

        # Temporal consistency for predictions
        self.face_history = []  # Stores (name, confidence) for recent frames

        # Face verification during capture
        self.capture_verification = []  # Stores (id, confidence) for duplicate checks

        # UI Layout
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Left: Video Feed
        self.video_frame = tk.Frame(self.main_frame, width=400, height=400)
        self.video_frame.pack(side="left", padx=10, pady=10)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Right: Controls
        self.control_frame = tk.Frame(self.main_frame)
        self.control_frame.pack(side="right", padx=10, pady=10, fill="y")

        tk.Label(self.control_frame, text="Attendance System", font=("Arial", 14, "bold")).pack(pady=10)

        # Add User
        tk.Label(self.control_frame, text="Enter Name:").pack()
        self.name_entry = tk.Entry(self.control_frame)
        self.name_entry.pack(pady=5)
        tk.Button(self.control_frame, text="Add New User", command=self.start_capture).pack(pady=5)

        # Delete User
        tk.Label(self.control_frame, text="Select User to Delete:").pack()
        self.delete_user_var = tk.StringVar()
        self.delete_user_combo = ttk.Combobox(self.control_frame, textvariable=self.delete_user_var, state="readonly")
        self.update_user_list()
        self.delete_user_combo.pack(pady=5)
        tk.Button(self.control_frame, text="Delete User", command=self.delete_user).pack(pady=5)

        # Show Attendance
        tk.Button(self.control_frame, text="Show Attendance", command=self.show_attendance).pack(pady=5)
        tk.Button(self.control_frame, text="Exit", command=self.exit_app).pack(pady=5)

        # Start camera
        self.start_camera()

    def load_names(self):
        self.names = {}
        if os.path.exists("names.pkl"):
            with open("names.pkl", "rb") as f:
                while True:
                    try:
                        name_dict = pickle.load(f)
                        self.names.update(name_dict)
                    except EOFError:
                        break

    def update_user_list(self):
        self.delete_user_combo['values'] = list(self.names.values())
        if self.names:
            self.delete_user_var.set(list(self.names.values())[0])
        else:
            self.delete_user_var.set("")

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Cannot open webcam")
                logging.error("Failed to open webcam")
                return
            for _ in range(5):
                self.cap.read()

        self.camera_running = True
        if os.path.exists(f"{trainer_path}/trainer.yml"):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                self.recognizer.read(f"{trainer_path}/trainer.yml")
            except Exception as e:
                logging.error(f"Error loading trainer.yml: {e}")
                self.recognizer = None

        self.update_video()

    def start_capture(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            logging.error("Empty name entered for capture")
            return
        existing_names = [n.lower() for n in self.names.values()]
        if name.lower() in existing_names:
            messagebox.showerror("Error", f"Name '{name}' already exists (case-insensitive)")
            logging.error(f"Duplicate name attempt: {name}")
            return

        if not self.recognizer:
            messagebox.showinfo("Info", "No trained model exists. Proceeding with new user capture.")
        self.capturing = True
        self.capture_count = 0
        self.capture_user_id = self.user_id
        self.capture_name = name
        self.capture_verification = []
        self.capture_cancelled = False
        logging.info(f"Started capture for user: {name}")
        messagebox.showinfo("Info", "Capturing 30 face images. Ensure good lighting and face the camera. Press 'q' to stop early.")

    def update_video(self):
        if not self.camera_running:
            return

        try:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image")
                logging.error("Failed to capture frame")
                self.camera_running = False
                return

            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 8, minSize=(50, 50))

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)

            best_name = None
            best_confidence = -1
            best_rect = None

            for (x, y, w, h) in faces:
                if self.capturing and self.capture_count < 30 and not self.capture_cancelled:
                    self.capture_count += 1
                    cv2.imwrite(f"{dataset_path}/User.{self.capture_user_id}.{self.capture_count}.jpg", gray[y:y+h, x:x+w])
                    cv2.putText(frame, f"Capturing {self.capture_count}/30", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if self.recognizer:
                        id_, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                        confidence_score = round(100 - confidence)
                        logging.info(f"Verification frame {self.capture_count}: ID {id_}, Confidence {confidence_score}%")
                        if confidence_score >= 50:
                            self.capture_verification.append((id_, confidence_score))
                            # Check for any 5 consecutive frames with same ID
                            if len(self.capture_verification) >= 5:
                                recent_ids = [id_ for id_, _ in self.capture_verification[-5:]]
                                id_counts = {id_: recent_ids.count(id_) for id_ in set(recent_ids)}
                                for id_, count in id_counts.items():
                                    if count >= 5:
                                        existing_name = self.names.get(id_, "Unknown")
                                        self.capture_cancelled = True
                                        self.root.after(10, lambda: self.show_duplicate_error(existing_name))
                                        self.reset_capture()
                                        break
                                if self.capture_cancelled:
                                    break

                    if self.capture_count >= 30:
                        if self.verify_captured_images():
                            self.finish_capture()
                        else:
                            self.root.after(10, lambda: self.show_duplicate_error("an existing user"))
                            self.reset_capture()
                        break

                elif not self.capturing and self.recognizer:
                    id_, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    confidence_score = round(100 - confidence)
                    if confidence_score < 50:
                        name = "Unknown"
                    else:
                        name = self.names.get(id_, "Unknown")

                    if confidence_score > best_confidence:
                        best_name = name
                        best_confidence = confidence_score
                        best_rect = (x, y, w, h)

            if best_name is not None:
                x, y, w, h = best_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                confidence_text = f"{best_name} {best_confidence:.2f}%"
                cv2.putText(frame, confidence_text, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if best_name != "Unknown" and best_confidence >= 60 and not self.capturing:
                    self.face_history.append((best_name, best_confidence))
                    if len(self.face_history) > 5:
                        self.face_history.pop(0)

                    name_counts = [n for n, _ in self.face_history]
                    if len(name_counts) >= 3 and name_counts[-3:].count(best_name) == 3:
                        if best_name in self.new_user_lock and (time.time() - self.new_user_lock[best_name]) < 30:
                            cv2.putText(frame, "New user: Wait 30s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                        elif self.check_attendance_today(best_name):
                            cv2.putText(frame, "Already Marked Today", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        else:
                            self.record_attendance(best_name)
                            cv2.putText(frame, "Attendance Marked", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 300), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk

            processing_time = (time.time() - start_time) * 1000
            logging.info(f"Frame processed in {processing_time:.2f}ms, Faces detected: {len(faces)}, Best name: {best_name}")

        except Exception as e:
            logging.error(f"Error in update_video: {e}")
            messagebox.showerror("Error", f"Video update failed: {e}")
            self.camera_running = False
            return

        if self.capturing and cv2.waitKey(1) & 0xFF == ord('q'):
            self.reset_capture()

        self.root.after(10, self.update_video)

    def show_duplicate_error(self, existing_name):
        messagebox.showerror("Error", f"Face matches existing user: {existing_name}. Registration cancelled.")
        logging.warning(f"Duplicate face detected: {existing_name}")

    def verify_captured_images(self):
        """Final verification of captured images against existing dataset."""
        if not self.recognizer:
            return True

        logging.info("Performing final verification of captured images")
        matches = []
        for i in range(1, self.capture_count + 1):
            img_path = f"{dataset_path}/User.{self.capture_user_id}.{i}.jpg"
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    id_, confidence = self.recognizer.predict(img)
                    confidence_score = round(100 - confidence)
                    if confidence_score > 75:
                        matches.append(id_)

        if matches:
            id_counts = {id_: matches.count(id_) for id_ in set(matches)}
            max_count = max(id_counts.values())
            if max_count >= 5:  # 5/30 images match
                id_ = max(id_counts, key=id_counts.get)
                logging.warning(f"Final verification failed: {max_count} images matched ID {id_}")
                return False

        # Temporary model check
        if not self.temporary_model_check():
            logging.warning("Temporary model check failed: face matches existing user")
            return False

        return True

    def temporary_model_check(self):
        """Train a temporary model with new images and test against existing dataset."""
        temp_recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces = []
        ids = []

        # Load new user's images
        for i in range(1, self.capture_count + 1):
            img_path = f"{dataset_path}/User.{self.capture_user_id}.{i}.jpg"
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    ids.append(self.capture_user_id)

        if not faces:
            return True

        try:
            temp_recognizer.train(faces, np.array(ids))
        except Exception as e:
            logging.error(f"Error training temporary model: {e}")
            return True

        # Test against existing users' images
        existing_images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') and f"User.{self.capture_user_id}." not in f]
        matches = []
        for img_path in existing_images:
            img = cv2.imread(os.path.join(dataset_path, img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                try:
                    id_, confidence = temp_recognizer.predict(img)
                    confidence_score = round(100 - confidence)
                    if confidence_score > 75:
                        matches.append(id_)
                except:
                    continue

        if matches:
            id_counts = {id_: matches.count(id_) for id_ in set(matches)}
            max_count = max(id_counts.values())
            if max_count >= 5:  # Significant matches
                return False

        return True

    def reset_capture(self):
        logging.info("Resetting capture state")
        for i in range(1, self.capture_count + 1):
            image_path = f"{dataset_path}/User.{self.capture_user_id}.{i}.jpg"
            if os.path.exists(image_path):
                os.remove(image_path)
        self.capturing = False
        self.capture_count = 0
        self.capture_user_id = None
        self.capture_name = None
        self.capture_verification = []
        self.capture_cancelled = False

    def finish_capture(self):
        logging.info(f"Finishing capture for user: {self.capture_name}")
        if self.capture_count >= 30:
            self.new_user_lock[self.capture_name] = time.time()
            with open("names.pkl", "ab") as f:
                pickle.dump({self.capture_user_id: self.capture_name}, f)
            self.user_id += 1
            with open("user_id.pkl", "wb") as f:
                pickle.dump(self.user_id, f)
            self.load_names()
            self.update_user_list()
            self.train_model_auto()
            messagebox.showinfo("Success", f"User {self.capture_name} added and model updated")
        else:
            messagebox.showerror("Error", "Not enough images captured")
        self.reset_capture()

    def delete_user(self):
        selected_name = self.delete_user_var.get()
        if not selected_name:
            messagebox.showerror("Error", "No user selected")
            return

        if not messagebox.askyesno("Confirm", f"Are you sure you want to delete {selected_name}?"):
            return

        user_id = None
        for id_, name in self.names.items():
            if name == selected_name:
                user_id = id_
                break

        if user_id is None:
            messagebox.showerror("Error", "User not found")
            return

        for i in range(1, 31):
            image_path = f"{dataset_path}/User.{user_id}.{i}.jpg"
            if os.path.exists(image_path):
                os.remove(image_path)

        if selected_name in self.new_user_lock:
            del self.new_user_lock[selected_name]

        new_names = {k: v for k, v in self.names.items() if k != user_id}
        with open("names.pkl", "wb") as f:
            for id_, name in new_names.items():
                pickle.dump({id_: name}, f)
        self.names = new_names
        self.update_user_list()

        self.train_model_auto()
        messagebox.showinfo("Success", f"User {selected_name} deleted and model updated")

    def train_model_auto(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = self.get_images_and_labels(dataset_path)

        if len(faces) > 0:
            recognizer.train(faces, np.array(ids))
            recognizer.write(f"{trainer_path}/trainer.yml")
            self.recognizer = recognizer

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faces = []
        ids = []

        for image_path in image_paths:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            id_ = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(img)
            ids.append(id_)

        return faces, ids

    def check_attendance_today(self, name):
        if not os.path.exists(attendance_file):
            return False

        today = datetime.now().strftime("%Y-%m-%d")
        try:
            df = pd.read_csv(attendance_file)
            return not df[(df["Name"] == name) & (df["Date"] == today)].empty
        except:
            return False

    def record_attendance(self, name):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if not os.path.exists(attendance_file):
            with open(attendance_file, "w") as f:
                f.write("Name,Date,Time\n")

        with open(attendance_file, "a") as f:
            f.write(f"{name},{date_str},{time_str}\n")

# Send signal to Arduino to blink LED
        if hasattr(self, 'arduino') and self.arduino:
            try:
                self.arduino.write(b'1')  # Send signal=
            except:
                logging.error("Failed to send signal to Arduino")


    def show_attendance(self):
        if not os.path.exists(attendance_file):
            messagebox.showerror("Error", "No attendance records found")
            return

        df = pd.read_csv(attendance_file)
        top = tk.Toplevel(self.root)
        top.title("Attendance Records")
        top.geometry("600x400")

        frame = tk.Frame(top)
        frame.pack(fill="both", expand=True, padx=5, pady=5)

        tree = ttk.Treeview(frame, columns=("Name", "Date", "Time"), show="headings", selectmode="browse")
        tree.heading("Name", text="Name")
        tree.heading("Date", text="Date")
        tree.heading("Time", text="Time")
        tree.pack(side="top", fill="both", expand=True)

        for idx, row in df.iterrows():
            tree.insert("", "end", iid=idx, values=(row["Name"], row["Date"], row["Time"]))

        button_frame = tk.Frame(frame)
        button_frame.pack(side="bottom", fill="x", pady=5)

        tk.Button(button_frame, text="Delete Selected", command=lambda: self.delete_selected_attendance(tree, df)).pack(side="left", padx=5)
        tk.Button(button_frame, text="Delete All", command=lambda: self.delete_all_attendance(tree)).pack(side="left", padx=5)

    def delete_selected_attendance(self, tree, df):
        selected_item = tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select a record to delete")
            return

        idx = int(selected_item[0])
        if not messagebox.askyesno("Confirm", "Are you sure you want to delete this attendance record?"):
            return

        df = df.drop(index=idx).reset_index(drop=True)
        df.to_csv(attendance_file, index=False)
        tree.delete(selected_item)

        for i, item in enumerate(tree.get_children()):
            tree.delete(item)
        for idx, row in df.iterrows():
            tree.insert("", "end", iid=idx, values=(row["Name"], row["Date"], row["Time"]))

        messagebox.showinfo("Success", "Attendance record deleted")

    def delete_all_attendance(self, tree):
        if not messagebox.askyesno("Confirm", "Are you sure you want to delete all attendance records?"):
            return

        with open(attendance_file, "w") as f:
            f.write("Name,Date,Time\n")

        for item in tree.get_children():
            tree.delete(item)

        messagebox.showinfo("Success", "All attendance records deleted")

    def exit_app(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceSystem(root)
    root.mainloop()