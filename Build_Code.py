import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Function to load images from a directory
def load_images(path):
    images = []
    classNames = []
    mylist = os.listdir(path)
    
    for cl in mylist:
        curImgPath = os.path.join(path, cl)
        curImg = cv2.imread(curImgPath)
        
        if curImg is None:
            print(f"Error: Unable to load image '{curImgPath}'")
            continue
        
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    
    return images, classNames

# Function to find face encodings from loaded images
def findEncodings(images):
    encodeList = []
    
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    
    return encodeList

# Function to mark attendance in a CSV file
def markAttendance(name):
    with open("S:\Projects\Face_Recognisation_Attendance\Face_Recognisation_Attendance-main\Attendance.csv", "r+") as f:
        myDataList = f.readlines()
        nameList = [line.split(",")[0] for line in myDataList]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'\n{name},{dtString}')

# Main script to capture video from webcam and perform face recognition
def main():
    path = "S:\Projects\Face_Recognisation_Attendance\Face_Recognisation_Attendance-main\ImagesAttendance"
    images, classNames = load_images(path)
    
    if not images:
        print("No images found. Please add images to the 'ImagesAttendance' directory.")
        return
    
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()  
        
        if not success:
            print("Failed to capture image from webcam.")
            break
        
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(name)
                top, right, bottom, left = faceLoc

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                markAttendance(name)

        cv2.imshow('Webcam', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

