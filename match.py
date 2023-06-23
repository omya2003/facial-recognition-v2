import csv
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime
#from ttest import test

filename = 'encodings.csv'
face_name = 'MS Dhoni'
img = cv2.imread('img_1.png')

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()


        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')

# Read the CSV file and retrieve the encoding for the specified face name
with open(filename, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['Name'] == face_name:
            known_encoding_str = row['Encoding']
            break
    else:
        print(f"Face name '{face_name}' not found in the CSV file.")
        known_encoding_str = None

if known_encoding_str is not None:
    known_encoding = np.fromstring(known_encoding_str[1:-1], dtype=float, sep=' ')
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_to_test_encoding = face_recognition.face_encodings(imgS)[0]
    results = face_recognition.compare_faces([known_encoding], image_to_test_encoding)
    #print(results)

if results[0] == True:
    print("Match!!!")

    # See how far apart the test image is from the known faces
    face_distances = face_recognition.face_distance([known_encoding], image_to_test_encoding)
    for i, face_distance in enumerate(face_distances):
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print(
            "- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(
            face_distance < 0.5))
        print()

    facesCurFrame = face_recognition.face_locations(img)
    #print(facesCurFrame)
    for faceCoordinates in facesCurFrame:
        y1, x2, y2, x1 = faceCoordinates
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, face_name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    now = datetime.now()
    print(f'Matched with : {face_name}, Time : {now} ')

    cv2.imshow('client', img)
    cv2.waitKey(2000)
    markAttendance(face_name)
    exit()

else:
    print("Mismatch!!!")

    facesCurFrame = face_recognition.face_locations(img)
    # print(facesCurFrame)
    for faceCoordinates in facesCurFrame:
        y1, x2, y2, x1 = faceCoordinates
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, 'unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('client', img)
    cv2.waitKey(5000)

    mismatch_folder = "mismatch"
    if not os.path.exists(mismatch_folder):
        os.makedirs(mismatch_folder)
    mismatch_filename = os.path.join(mismatch_folder, face_name + '_mismatch.jpg')
    cv2.imwrite(mismatch_filename, img)

    exit()






