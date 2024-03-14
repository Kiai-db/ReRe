from segment_local import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image



def is_contour_smooth(contour, threshold_angle):
    # Function to check if a contour has any sharp angles
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        p3 = contour[(i + 2) % len(contour)][0]

        # Calculate the angle between the points
        angle = np.abs(np.rad2deg(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])))
        if angle < threshold_angle:
            return False  # Sharp angle detected
    return True

def save_cropped_objects(masks, original_image, output_folder):
    image_array = []  # Initialize an empty list to store the PIL Images
    for i, ann in enumerate(masks):
        mask = ann['segmentation']  # This should be a 2D numpy array

        # Skip small masks
        if mask.sum() < 5000:  # Adjust the threshold as needed
            continue

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Assume fruits do not have sharp angles, less than x degrees
        if not all(is_contour_smooth(contour, 80) for contour in contours):
            continue
        
        # Find the bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the mask and image
        mask_cropped = mask[rmin:rmax, cmin:cmax]
        image_cropped = original_image[rmin:rmax, cmin:cmax]

        # Create an RGBA image with a transparent background
        # Only the object itself will be opaque
        output_image = np.zeros((rmax-rmin, cmax-cmin, 4), dtype=np.uint8)
        output_image[..., :3] = image_cropped
        output_image[..., 3] = mask_cropped * 255  # Alpha channel

        # Convert to PIL Image and add to the list
        pil_image = Image.fromarray(output_image)
        image_array.append(pil_image)

    return image_array  # Return the list of PIL Images


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


model_type = "vit_l"  #vit_b - 2:10 min, vit_l - 2:45 min, vit_h - 3:30 min
checkpoint_path = "CNNs\sam_vit_l_0b3195.pth" 

def getobjects(image): 
    print("check0")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    print("check1")

    mask_generator = SamAutomaticMaskGenerator(sam)
    print("check2")
    masks = mask_generator.generate(image)
    print("check3")

    # Define the directory where you want to save the images
    output_folder = "Crop_Fruit\croppedobjects"
    imarray = save_cropped_objects(masks, image, output_folder)


    print("All objects saved as individual images.")
    return imarray