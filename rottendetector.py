import torch
from torchvision import transforms, models
from PIL import Image
import os

def rottenCNN(dataset_path, model_load_path, results_file_path):
    """
    Run image classification on a dataset of images, saving predictions to a file.
    
    Parameters:
    - dataset_path: Path to the folder containing images.
    - model_load_path: Path to the saved model file.
    - results_file_path: Path where the predictions text file will be saved.
    """
    
    transform = transforms.Compose([
        transforms.Lambda(lambda image: image.convert('RGB')),  
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the model
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)

    # Load the model 
    model.load_state_dict(torch.load(model_load_path))

    # Use CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Prepare the dataset
    image_paths = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Prediction and logging
    with open(results_file_path, 'w') as results_file:
        for image_path in image_paths:
            image = Image.open(image_path)
            image = transform(image).unsqueeze(0)
            
            if torch.cuda.is_available():
                image = image.cuda()
            
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                predicted_class = preds.item()
            
            # Write the image name and its predicted class to the text file
            results_file.write(f"{os.path.basename(image_path)}: {predicted_class}\n")

    print("Image classification complete. Results saved to predictions.txt.")
