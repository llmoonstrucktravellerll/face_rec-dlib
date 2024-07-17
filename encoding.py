import cv2
import face_recognition
import pickle
import os

#importing the faces
folderPath = 'images'
imagePathList = os.listdir(folderPath)
imgList = []
name = []
for path in imagePathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    name.append(os.path.splitext(path)[0])
# print(len(imgList))
# print(name)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print("Encoding Start")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithNames = [encodeListKnown, name]
print("Encoding Complete")

file = open('encodingcas.p', 'wb')
pickle.dump(encodeListKnownWithNames, file)
file.close()