import csv
import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime
from ttest import test
import socket

# Server configuration
host = '0.0.0.0'  # Server IP address
port = 5000  # Server port number

filename = 'encodings.csv'
face_name = 'Omi'

# Create a TCP/IP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Bind the socket to the host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(1)

print('Server listening on {}:{}'.format(host, port))

while True:
    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    print('Client connected:', client_address)

    # Receive image data from the client
    image_data = b''
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        image_data += data

    # Convert image data to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


#img = cv2.imread('Screenshot from 2023-06-22 11-11-53.png')
if img is None:
    print("Error: Failed to load the image.")
    exit()



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

    #desired_width = 800
    #desired_height = int(desired_width * 3 / 4)
    #imgS = cv2.resize(img, (desired_width, desired_height))
    #imgS = cv2.resize(img, (800, 600))


    imgS = cv2.resize(img, (800, 600), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    label = test(image=imgS,
                 model_dir='/media/omi/a1/facerecog/Facial-Recognition-w-spoofing-detection/Silent-Face-Anti-Spoofing-master/resources/anti_spoof_models',
                 device_id=0,
                 )
    if label == 1:
        image_to_test_encoding = face_recognition.face_encodings(imgS)[0]
        results = face_recognition.compare_faces([known_encoding], image_to_test_encoding)
        # print(results)

    else:
        print("Image captured is not live. Please try Again.")
        exit()


if results[0] == True:

    print("        ███╗░░░███╗░█████╗░████████╗░█████╗░██╗░░██╗██╗")
    print("        ████╗░████║██╔══██╗╚══██╔══╝██╔══██╗██║░░██║██║")
    print("        ██╔████╔██║███████║░░░██║░░░██║░░╚═╝███████║██║")
    print("        ██║╚██╔╝██║██╔══██║░░░██║░░░██║░░██╗██╔══██║╚═╝")
    print("        ██║░╚═╝░██║██║░░██║░░░██║░░░╚█████╔╝██║░░██║██╗")
    print("        ╚═╝░░░░░╚═╝╚═╝░░╚═╝░░░╚═╝░░░░╚════╝░╚═╝░░╚═╝╚═╝")

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
        #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
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
    face_distances = face_recognition.face_distance([known_encoding], image_to_test_encoding)
    for i, face_distance in enumerate(face_distances):
        print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
        print("- With a normal cutoff of 0.6, would the test image match the known image? {}".format(face_distance < 0.6))
        print("- With a very strict cutoff of 0.5, would the test image match the known image? {}".format(face_distance < 0.5))
        print()

    facesCurFrame = face_recognition.face_locations(img)
    # print(facesCurFrame)
    for faceCoordinates in facesCurFrame:
        y1, x2, y2, x1 = faceCoordinates
        #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
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

# Send the result back to the client
    #result = "Recognition result"   Replace with your own result
    client_socket.sendall(result.encode())

    # Close the connection
    client_socket.close()

# Close the server socket
server_socket.close()




