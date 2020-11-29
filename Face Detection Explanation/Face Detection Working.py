import cv2

import numpy

import face_recognition

# Step 1:-
# We will convert BGR images to RGB 

imgElon = face_recognition.load_image_file("Images Basic/Elon.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file("Images Basic/Bill Gates.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)


# Step 2:-
# Finding the Faces in the Images along with their Encoding

# In the line of code given below we are sending the first element ( [0] ) as we specify only one image to face_location

faceLoc = face_recognition.face_locations(imgElon)[0] 
# print(faceLoc)
# Above print staement returns a tuple containing 4 values which are:- Top, Right, Bottom and Left
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, ( faceLoc[3], faceLoc[0] ), ( faceLoc[1], faceLoc[2] ), (255, 0, 255), 2)

faceLoctest = face_recognition.face_locations(imgTest)[0] 
encode_test = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, ( faceLoctest[3], faceLoctest[0] ), ( faceLoctest[1], faceLoctest[2] ), (255, 0, 255), 2)

# cv2. rectangle takes 4 arguments :-
# 1. Image
# 2. Coordinates of face as :- (Left, Top) , (Right, Bottom)
# 3. Color :- RGB (255, 0, 255) i.e. Purple in this case
# 4. Thickness
 
# Step 3:-
# We will mow compare these faces and find the distance between them

results = face_recognition.compare_faces([encodeElon] ,encode_test)
print(results)
# In the above example, print statement returns True which means both encodings are similar

# Also, the first argument of compare_faces is that it takes a lis tof faces we want our machine to learn (somewhat
# like training data ) while the second argument takes the test data.
# If results return True for different pics with the faces of the same person, then we our program is working fine

face_distance = face_recognition.face_distance([encodeElon], encode_test)
# Lower the value of distance , better is the match
print(face_distance)

# Now we will display the results in the test image and tell the original image

cv2.putText(imgTest, f'{results[0]} {round(face_distance[0], 2)}', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.putText(imgElon, f'Elon Musk', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 0, 255), 2)




cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Test Image', imgTest)

cv2.waitKey(0)
