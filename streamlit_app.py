from Segment_Ingr import getobjects  # Make sure this function is properly defined
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
# Load the model
model = load_model('ReReCNN/CNNs/ReReCNNv2')
image_path = "Images/base_image.jpg"  
your_image = cv2.imread(image_path)
your_image = cv2.cvtColor(your_image, cv2.COLOR_BGR2RGB)  
# Get the objects (PIL Images) from the original image
pil_images = getobjects(your_image)  # Ensure this returns a list of PIL Image objects

# Initialize an empty list to store tuples of (PIL Image, class_name)
classified_objects = []

# Define your class names
class_names = [
    "apple", "banana", "beetroot", "bell pepper", "cabbage", "capsicum", "carrot", "cauliflower", "chili pepper", "corn",
    "cucumber", "eggplant", "garlic", "ginger", "grapes", "jalapeno", "kiwi", "lemon", "lettuce", "mango", "onion", "orange", "paprika", "pear", "peas",
    "pineapple", "pomegranate", "potato", "radish", "soy beans", "spinach", "sweetcorn", "sweetpotato", "tomato", "turnip",
    "unwanted", "watermelon"
]

# Process each PIL Image for classification
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

    # Append the PIL image and class name to the list if the class is not "unwanted"
    if predicted_class_name != "unwanted":
        classified_objects.append((pil_image, predicted_class_name))
# classified_objects now contains tuples of (PIL Image, predicted class name)
# for all objects classified as something other than "unwanted"
