import cv2
import numpy as np
from tensorflow import keras

model = keras.models.load_model('../apple_latest.h5')

# Function to preprocess the image before classification
def preprocess_image(image):
    image = cv2.resize(image, (128, 128)) 
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) 
    return image


def classify_leaf(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return prediction

# Open your computer's camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame from the camera.")
        break

    # Perform classification
    prediction = classify_leaf(frame)

    # Define disease classes (you should adapt this to your specific classes)
    disease_classes = ["BLOTCH", "NORMAL", "ROT", "SCAB"]

    # Get the predicted class
    predicted_class_index = np.argmax(prediction)
    predicted_class = disease_classes[predicted_class_index]

    # Display the result
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Apple Disease Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
