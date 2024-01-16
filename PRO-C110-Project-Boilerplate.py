import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("keras_model.h5")

# Load labels
with open('labels.txt', 'r') as file:
    class_labels = [label.strip() for label in file.readlines()]

# Define a video capture object
vid = cv2.VideoCapture(0)

while True:
    # Capture the video frame by frame
    ret, frame = vid.read()

    # Flip the frame if needed (adjust flipCode as per your requirement)
    frame = cv2.flip(frame, 1)

    # Resize the frame to match the model input size
    resized_frame = cv2.resize(frame, (224, 224))

    # Expand dimensions to match the model's expected input shape
    input_frame = np.expand_dims(resized_frame, axis=0)

    # Normalize the frame
    normalized_frame = input_frame / 255.0

    # Make a prediction
    prediction = model.predict(normalized_frame)

    # Get the predicted class label
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the predicted class on the frame
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Rock Paper Scissors Classifier', frame)

    # Check for the spacebar key to exit the loop
    key = cv2.waitKey(1)
    if key == 32:
        break

# Release the video capture object
vid.release()

# Close all open windows
cv2.destroyAllWindows()
