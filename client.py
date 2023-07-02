import socket
import cv2
import numpy as np

# Server configuration
server_host = '0.0.0.0'  # Server IP address
server_port = 5000  # Server port number

# Capture image using the camera
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
if not ret:
    print("Error: Failed to capture image.")
    camera.release()
    exit()

# Convert the captured image to JPEG format
_, img_encoded = cv2.imencode('.jpg', frame)

# Create a TCP/IP socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the server
    client_socket.connect((server_host, server_port))

    # Send the image data to the server
    client_socket.sendall(img_encoded.tobytes())

    # Receive the result from the server
    result = client_socket.recv(1024).decode()
    print('Result:', result)

finally:
    # Close the client socket
    client_socket.close()

# Release the camera
camera.release()
