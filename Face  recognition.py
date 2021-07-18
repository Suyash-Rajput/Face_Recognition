import numpy as np
import cv2
import os
import face_recognition

path ='ISJ'
print(path)
mylist =os.listdir(path)
print(mylist)
images =[]
classname = []
for c1 in mylist:
    curImg = cv2.imread(f'{path}/{c1}')
    images.append(curImg)
    classname.append(os.path.splitext(c1)[0])
print(classname)


def encoding(images):
    encodelist = []
    for img in images:
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist
encodinglistknown = encoding(images)
print(encodinglistknown)

print('Encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    img_Reduction =cv2.resize(img,(0,0),None,0.25,0.25)
    img_Reduction=cv2.cvtColor(img_Reduction,cv2.COLOR_BGR2RGB)
    face_frame =face_recognition.face_locations(img_Reduction)
    encode_frame = face_recognition.face_encodings(img_Reduction,face_frame)


    for encodeface,faceloc in zip(encode_frame,face_frame):
        matches= face_recognition.compare_faces(encodinglistknown,encodeface)
        faceDis= face_recognition.face_distance(encodinglistknown,encodeface)
        print(faceDis)
        matchIndex= np.argmin(faceDis)

        if matches[matchIndex]:
            name= classname[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 =faceloc
            y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y1-35),(x2,y2),(0,255,0),2)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

