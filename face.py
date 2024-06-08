import cv2

# Trained dataset
trainedDataset = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#read a Image
img = cv2.imread('images/group-3.jpg')

# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = trainedDataset.detectMultiScale(gray)
print(faces)

for x, y, w, h in faces:
    cv2.rectangle(img, (x,y), (x + w, y + h), (255, 0 ,0), 2)

cv2.imshow('Matt', img)
# cv2.imshow('Gray', gray)
cv2.waitKey()



