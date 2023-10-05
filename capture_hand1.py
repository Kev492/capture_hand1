from keras.models import load_model
import cv2
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/pyworkspace1/camera_test/keras_sample/converted_keras_second/keras_Model.h5", compile=False)

# Load the labels
class_names = open("C:/pyworkspace1/camera_test/keras_sample/converted_keras_second/labels.txt", "r").readlines()

output_file_path = "C:/pyworkspace1/camera_test/capture_hand/sign1.txt"

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

# Show the image in a window
cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)

start_time = time.time()  # 타이머 시작 시간 저장

while True:
    # Grab the web camera's image
    ret, image = camera.read()

    if ret:
        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

    # Check if 2 seconds have elapsed
    if time.time() - start_time > 2:
        # Make the image a numpy array and reshape it to the model's input shape
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predict the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:])
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

        # Write prediction and confidence score to the output file
        with open(output_file_path, "w") as output_file:
             output_file.write(class_name[2:])
             output_file.write(str(np.round(confidence_score * 100))[:-2] + "%" + "\n")

        break

# Release the camera and close the OpenCV windows
camera.release()
cv2.destroyAllWindows()