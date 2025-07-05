# Fast-Track
# Face Detection Attendance System

A real-time face recognition attendance system built using **OpenCV**, **Tkinter**, and **Arduino**. This project detects and recognizes faces from a webcam feed, marks attendance with timestamps, and optionally triggers Arduino-based hardware (e.g., LED indicator)

---

## 🛠 Features

- Real-time face detection and recognition
- New user registration with duplicate face prevention
- Attendance logging with date and time
- GUI interface using Tkinter
- View and manage attendance records
- Delete user data and retrain model
- Optional Arduino support for hardware triggers (e.g., LED blink)

---

## 📸 Technologies Used

- Python 3.x
- OpenCV (`opencv-contrib-python`)
- Tkinter (standard GUI library)
- Pillow (for image rendering)
- Pandas (CSV handling)
- Arduino (via `pyserial` for serial communication)

---

## 🧱 Directory Structure

Face-Attendance-System/
├── dataset/ # Captured face images
├── trainer/ # Trained model file (trainer.yml)
├── attendance.csv # Attendance records
├── names.pkl # Pickled dictionary of user IDs and names
├── user_id.pkl # Tracks next user ID
├── haarcascade_frontalface_default.xml # Haar Cascade for face detection
├── main.py # Main application script
└── README.md # Project documentation

👤 Adding a New User
>Enter the user's name in the text field.

>Click "Add New User".

>The system will capture 30 face images and train the model.

>Duplicates are detected using confidence scores and live validation.

📅 Viewing Attendance
>Click "Show Attendance" to view all records.

>You can delete individual or all records from within the GUI.

🗑 Deleting a User
>Select a name from the dropdown and click "Delete User".

>Their images and training data will be removed and the model will retrain automatically.

📌 Notes
>Face data and user names are stored using pickle.

>Attendance is stored in attendance.csv.

>The app locks repeated attendance marking for the same user within 30 seconds.

📷 Screenshots
>Add screenshots here showing:

>The GUI

>Real-time face recognition

>Attendance window

🧠 Future Improvements
>Cloud database integration (Firebase / MongoDB)

>Mask detection support

>Role-based access

>Deploy as a desktop app with PyInstaller

