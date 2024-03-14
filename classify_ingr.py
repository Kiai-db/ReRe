from Segment_Ingr import getobjects  # Make sure this function is properly defined
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
# Load the model
model = load_model('CNNs\ReReCNN.h5')
classified_objects = []
# Define your class names
class_names = [
    "apple", "banana", "bell pepper","carrot", "cauliflower", "cucumber", "garlic", "ginger", "grapes",
    "kiwi", "lemon", "mango", "onion", "orange", "pear", "peas", "pineapple", "pomegranate",
    "potato", "sweetcorn", "tomato", "watermelon"
]

# Process each PIL Image for classification
def classify_ingr(pil_images):
    for pil_image in pil_images:
        pil_image = pil_image.convert('RGB')
        # Convert the PIL Image to a numpy array
        img_array = img_to_array(pil_image.resize((224, 224)))  # Resize the image to 224x224
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        # Predict the class
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        print(predicted_class_name)

        classified_objects.append((pil_image, predicted_class_name))
   
    return classified_objects