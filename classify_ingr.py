from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import shutil

# Load the model
model = load_model('CNNs\ReReCNNv7.keras')

# Define your class names
class_names = [
    "apple", "banana", "banana", "capsicum", "orange", "tomato"
]

def empty_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


# Define a confidence threshold
confidence_threshold = 0.90  # = 0.x, x%

# Process each PIL Image for classification
def classify_ingr(pil_images):
    output_folder = 'detected_objects'
    empty_folder(output_folder)
    classified_objects = []  # Initialize within the function to avoid using global variable
    num = 0
    for pil_image in pil_images:

        pil_image = pil_image.convert('RGB')
        # Convert the PIL Image to a numpy array
        img_array = img_to_array(pil_image.resize((224, 224)))  # Resize the image to 224x224
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Predict the class
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_confidence = np.max(predictions, axis=1)[0]
        print("pred: ", predicted_confidence)
        # Check if the prediction confidence is above the threshold
        if predicted_confidence >= confidence_threshold:
            predicted_class_name = class_names[predicted_class_index]
            print(f"Classified as {predicted_class_name} with confidence {predicted_confidence}")
            classified_objects.append((pil_image, predicted_class_name))
            image_path = os.path.join(output_folder, f"object_{num}_{predicted_class_name}.png")
            pil_image.save(image_path)
        else:
            # Here we append 'uncertain' label or skip appending depending on requirements
            print("Not confident enough to classify")
            #classified_objects.append((pil_image, 'uncertain'))  # or just continue without appending
            image_path = os.path.join(output_folder, f"object_{num}_uncertain.png")
            pil_image.save(image_path)

        num += 1
    return classified_objects

# Example usage:
# pil_images = getobjects("image.png")  # Ensure this returns a list of PIL Image objects
# classified_results = classify_ingr(pil_images)
