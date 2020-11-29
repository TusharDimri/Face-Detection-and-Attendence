import cv2

import numpy as np

import face_recognition

import os

path = "Attendence Images"

images = []
classNames = []
mylist = os.listdir(path)
# print(mylist)
for cl in mylist:
    current_Image = cv2.imread(f'{path}/{cl}')
    images.append(current_Image)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)

# Now we will create a function that will compute the encodings for us

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converting the Image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodings_known = find_encodings(images)

# print(len(encodings1  _known))

print("Encoding Complete")

# Now we will take image from webcam to find whether they match with known images or not

cap = cv2.VideoCapture(0)
while True:
    success, image = cap.read()
    imageSmall = cv2.resize(image, (0,0), None,  0.25, 0.25) # (0, 0) means we have not defined out pixel size
    # We will reduce the image size to one fourth its original size to speed up the process
    imageSmall = cv2.cvtColor(imageSmall, cv2.COLOR_BGR2RGB)

    facesLoc= face_recognition.face_locations(imageSmall)
    # We find locatiosn of the faces as we may find more than one face from webcam image
    encodingsFace = face_recognition.face_encodings(imageSmall, facesLoc)


    for encodeFace, faceLoc in zip(encodingsFace, facesLoc):
        matches = face_recognition.compare_faces(encodings_known, encodeFace)
        faceDis = face_recognition.face_distance(encodings_known, encodeFace)
        # print(faceDis, matches)

        # compare_faces will find all the matches of our webcam image with the images we have in the list and return a list containing True and False values
        # face_distance will find the distace of webcam image from known images and return a list containing those values

        # Now we will find the lowest distance (matches = True ) to find whether image from webcam matches from any image known  
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2 , y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Webcam', image)
            cv2.waitKey(1)
            

        else:
            name = 'Unknown'
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 =  y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(image, (x1, y2 - 35), (x2 , y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Webcam', image)
            cv2.waitKey(1)
       