import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import cv2
import numpy as np
import os

"""
Class 0: OverRipe
Class 1: Ripe
Class 2: Rotten
Class 3: UnRipe
"""

def setup_model(model_load_path):
    model = resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(model_load_path))
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model

def predict_and_log(model, image_paths, transform, results_file_path):
    with open(results_file_path, 'w') as results_file:
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            if torch.cuda.is_available():
                image = image.cuda()
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                predicted_class = preds.item()
            results_file.write(f"{os.path.basename(image_path)}: {predicted_class}\n")

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

def update_predictions_with_feature_detection(predictions_file_path, dataset_path):
    images_to_process = []
    with open(predictions_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(': ')
            if parts[1] in ['0', '2']:
                images_to_process.append(parts[0])
    bruise_percentage_results = {}
    for image_name in images_to_process:
        image_path = os.path.join(dataset_path, image_name)
        image = cv2.imread(image_path)
        bruise_mask = detect_features(image)
        bruise_area = calculate_area(bruise_mask)
        total_area = image.shape[0] * image.shape[1]
        bruise_percentage = (bruise_area / total_area) * 100
        bruise_percentage_results[image_name] = bruise_percentage
    with open(predictions_file_path, 'a') as file:
        for image_name, percentage in bruise_percentage_results.items():
            file.write(f"{image_name}: Bruise Percentage: {percentage:.2f}%\n")

def rottenCNN():
    # Paths and configurations
    dataset_path = "TestImages"
    model_load_path = "CNNs/model.pth"
    results_file_path = "predictions/predictions.txt"
    print("lol")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = setup_model(model_load_path)

    image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    predict_and_log(model, image_paths, transform, results_file_path)

    update_predictions_with_feature_detection(results_file_path, dataset_path)

    print("Image classification and feature detection complete. Results updated in predictions.txt.")


rottenCNN()