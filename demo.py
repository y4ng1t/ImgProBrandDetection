import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_path = 'path/to/your/saved_model'  # Update this with your actual path
model = tf.keras.models.load_model(model_path)

# Load the image from file
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure correct color format
    img = cv2.resize(img, (224, 224))  # Resize to match the model's expected sizing
    img = img / 255.0  # Normalize pixel values to between 0 and 1
    return img

# Perform brand detection on an image
def detect_brand(image_path):
    img = load_image(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class

# Upload and detect brands for a set of images
def process_images(image_paths):
    for image_path in image_paths:
        predicted_class = detect_brand(image_path)
        print(f"Image: {image_path}, Predicted Class: {predicted_class}")

# Example usage
image_paths = ['path/to/your/image1.jpg', 'path/to/your/image2.jpg', 'path/to/your/image3.jpg']
process_images(image_paths)
