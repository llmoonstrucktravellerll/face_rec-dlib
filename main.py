import cv2
import pickle
import face_recognition
import numpy as np
import threading

cv2.ocl.setUseOpenCL(True)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# load encodings

file = open('encodingcas.p', 'rb')
encodeListKnownWithNames = pickle.load(file)
file.close()
encodeListKnown, names = encodeListKnownWithNames
# print(names)

lock = threading.Lock()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        distance = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print("matches", matches)
        # print("distance", distance)
        matchIndex = np.argmin(distance)
        if matches[matchIndex] == True and distance[matchIndex] < 0.5:
            print(names[matchIndex])
            # Convert face location from scaled coordinates to original size
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Write name next to the face
            cv2.putText(img, names[matchIndex], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        else:
            print("Unknown")
            # Convert face location from scaled coordinates to original size
            top, right, bottom, left = faceLoc
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Write name next to the face
            cv2.putText(img, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

