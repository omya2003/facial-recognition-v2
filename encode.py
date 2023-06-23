import cv2
import face_recognition
import csv
import os

path = 'Training_images'
images = []
classNames = []
filename = 'encodings.csv'
myList = os.listdir(path)
print(myList)

# Load existing encodings from the CSV file
existingEncodings = {}
if os.path.isfile(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            name = row[0]
            encoding = [row[1:]]
            existingEncodings[name] = encoding
            #print(encoding)

# Iterate over each file in the list
for cl in myList:
    name = os.path.splitext(cl)[0]
    if name in existingEncodings:
        print(f"Skipping image '{name}' as it already exists in the CSV file.")
        continue

    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(name)

def findEncodings(images, classNames):
    encodeDict = {}

    for img, name in zip(images, classNames):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeDict[name] = encode

    return encodeDict

print(classNames)

# Find encodings only for new images
encodeDictKnown = findEncodings(images, classNames)
print(encodeDictKnown)

# Write the encodings to the CSV file
with open(filename, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for name, encoding in encodeDictKnown.items():
        writer.writerow([name] + [encoding])

