# Import the necessary libraries

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import json
import numpy as np
import time
import os

# Load a pre-trained YOLOv8 model
model_yolo = YOLO('yolov8n.pt')


# Function to create a JSON file
def create_json_post(frame, cls, frame_number):
    # Convert frame to list
    frame_in_arr = np.asarray(frame).tolist()
    data = {
        "type": cls,
        "frame": frame_in_arr,
        "time": frame_number  # Adding frame number as the time indicator
    }
    file_path = f"{time.strftime('%d_%m_%Y-%H_%M_%S')}.json"
    with open(file_path, "w") as json_file:
        json.dump(data, json_file)
    # Save the frame as an image
    cv2.imwrite(f"{os.path.splitext(file_path)[0]}-{cls}.jpg", frame)

    return file_path


# Function to detect objects in an image
def image_object_detection(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects in the image
    results = model_yolo(image_rgb)

    # Annotate the image with bounding boxes
    annotated_image = results[0].plot()

    # Check if any pedestrian is detected and create a JSON file
    for obj in results[0].boxes.data:
        if model_yolo.names[int(obj[5])] == 'person':
            create_json_post(image, 'person', 0)
            break

    # Display the annotated image
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis('on')
    plt.show()


# Function to process video frames
def detect_objects_in_video(video_path, output_path=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object if output_path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects in the frame
        results = model_yolo(frame_rgb)

        # Annotate the frame with bounding boxes
        annotated_frame = results[0].plot()

        # Check if any person is detected and create a JSON file
        for obj in results[0].boxes.data:
            if model_yolo.names[int(obj[5])] == 'person':
              create_json_post(frame, 'person', frame_number)
              break

        # Display the annotated frame
        cv2.imshow('YOLOv8s Object Detection', annotated_frame)

        # Write the frame to the output video if output_path is provided
        if output_path:
            out.write(annotated_frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer objects
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


# Function to process video frames from a camera
def detect_objects_from_camera(output_path=None):
    # Open the camera
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object if output_path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect objects in the frame
        results = model_yolo(frame_rgb)

        # Annotate the frame with bounding boxes
        annotated_frame = results[0].plot()

        # Check if any person is detected and create a JSON file
        for obj in results[0].boxes.data:
            if model_yolo.names[int(obj[5])] == 'person':
                create_json_post(frame, 'person', frame_number)
                break

        # Display the annotated frame
        cv2.imshow('YOLOv8s Object Detection', annotated_frame)

        # Write the frame to the output video if output_path is provided
        if output_path:
            out.write(annotated_frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
          print("Quitting...")
          break

    # Release video capture and writer objects
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()


# # Example usage for detecting objects in an image
# image_path = '1.jpg'  # Replace with your image path
# image_object_detection(image_path)
#
# # Example usage: Detect objects in a video and display the results
# video_path = '2.mp4'
# output_path = 'video_yolo_cust2.mp4'
# detect_objects_in_video(video_path, output_path)

# Example usage: Detect objects from camera and display the results
output_path_camera = 'camera_yolo_output.mp4'  # Optional: to save the output video from the camera
detect_objects_from_camera(output_path_camera)


