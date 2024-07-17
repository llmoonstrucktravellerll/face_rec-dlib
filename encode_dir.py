import cv2
import face_recognition
import pickle
import os

folderPath = 'images'

def findEncodings(folderPath):
    encodings = {}

    for person_dir in os.listdir(folderPath):
        person_path = os.path.join(folderPath, person_dir)
        if os.path.isdir(person_path):
            person_encodings = []
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)
                if len(encode) > 0:
                    person_encodings.append(encode[0])
            if person_encodings:
                encodings[person_dir] = person_encodings

    return encodings

print("Encoding Start")
encodings = findEncodings(folderPath)
print("Encoding Complete")

file = open('encodingdir.p', 'wb')
pickle.dump(encodings, file)
file.close()