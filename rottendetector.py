import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import cv2
import numpy as np
import os

# Define classes for readability
CLASS_LABELS = ["OverRipe", "Ripe", "Rotten", "UnRipe"]

def setup_model(model_load_path):
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_load_path))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

def predict_image(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = preds.item()
    return predicted_class

def detect_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bruise_range = [np.array([10, 50, 50]), np.array([30, 255, 255])]
    unripe_range = [np.array([30, 50, 50]), np.array([70, 255, 255])]
    unripe_mask = cv2.inRange(hsv_image, unripe_range[0], unripe_range[1])
    bruise_mask = cv2.inRange(hsv_image, bruise_range[0], bruise_range[1])
    not_unripe_mask = cv2.bitwise_not(unripe_mask)
    bruise_mask = cv2.bitwise_and(bruise_mask, not_unripe_mask)
    kernel = np.ones((5, 5), np.uint8)
    bruise_mask = cv2.morphologyEx(bruise_mask, cv2.MORPH_OPEN, kernel)
    return bruise_mask

def calculate_area(mask):
    return cv2.countNonZero(mask)

def process_predictions(model, image_paths, transform, dataset_path):
    results = []
    for image_path in image_paths:
        predicted_class = predict_image(model, image_path, transform)
        if predicted_class in [0, 2]:  # OverRipe or Rotten
            image = cv2.imread(image_path)
            bruise_mask = detect_features(image)
            bruise_area = calculate_area(bruise_mask)
            total_area = image.shape[0] * image.shape[1]
            bruise_percentage = (bruise_area / total_area) * 100
            results.append((image_path, CLASS_LABELS[predicted_class], bruise_percentage))
        else:
            results.append((image_path, CLASS_LABELS[predicted_class], 0))  # Ripe or UnRipe with 0% bruise
    return results

def rottenCNN():
    dataset_path = "TestImages"
    model_load_path = "CNNs/model.pth"
    results_file_path = "predictions/predictions.txt"

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = setup_model(model_load_path)

    image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = process_predictions(model, image_paths, transform, dataset_path)

# Logging the results
    with open(results_file_path, 'w') as results_file:
        for image_path, predicted_class, bruise_percentage in results:
            line = f"{os.path.basename(image_path)}: {predicted_class}"
            if bruise_percentage > 0:
                line += f", Bruise Percentage: {bruise_percentage:.2f}%\n"
            else:
                line += ", Bruise Percentage: 0%\n"  
            results_file.write(line)


    print("Image classification and feature detection complete. Results updated in predictions.txt.")

rottenCNN()
