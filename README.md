### Face Recognition and Attendance Marking Script

#### Overview
This Python script enables real-time face recognition using OpenCV and the `face_recognition` library. It captures video from the webcam, detects faces, compares them with known faces, and marks attendance based on recognized faces.

#### Features
- **Real-time Face Recognition:** The script continuously captures video from the webcam and performs face recognition on each frame in real-time.
- **Attendance Marking:** It marks attendance by appending recognized names and current date-time to a CSV file named "Attendance.csv".
- **Dynamic Encoding:** Faces from images in the "ImagesAttendance" directory are dynamically encoded for comparison during runtime.
- **User Interaction:** The program can be exited by pressing 'q', which is monitored using the `cv2.waitKey()` function.

#### Usage
1. Ensure that the "ImagesAttendance" directory contains images of individuals whose attendance needs to be marked.
2. Run the script, and it will open a window displaying the webcam feed with face recognition and attendance marking functionality.
3. Press 'q' to exit the program.

#### Dependencies
- OpenCV (cv2)
- NumPy (np)
- face_recognition
- os
- datetime

  # Face Recognition with OpenCV and face_recognition

This repository contains Python scripts for performing face recognition using OpenCV and the `face_recognition` library. It demonstrates how to detect and recognize faces in images.

## Features
- **Face Detection:** Detect faces in images using OpenCV's Haar cascades.
- **Face Encoding:** Encode detected faces into numerical representations using the `face_recognition` library.
- **Face Comparison:** Compare faces to determine if they belong to the same person.
- **Visualization:** Visualize the detected faces and recognition results with rectangles and text annotations.

## Scripts
1. **`face_comparison.py`:** Compares faces between two images and displays the result.
2. **`face_detection.py`:** Detects and visualizes faces in an image.
3. **`face_encoding.py`:** Encodes faces in an image into numerical representations.

## Usage
1. Clone the repository: `git clone https://github.com/your-username/face-recognition.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the desired script: `python script_name.py`

## Dependencies
- OpenCV
- NumPy
- face_recognition


