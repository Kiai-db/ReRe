from segment_local import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import imagehash
from PIL import Image


def rgba_to_rgb_white_bg(pil_image):
    rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
    rgb_image.paste(pil_image, mask=pil_image.split()[3])  # Split off the alpha channel and use as mask
    return rgb_image


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
    image_array = []
    image_hashes = set()  # To store hashes of images already processed

    for i, ann in enumerate(masks):
        mask = ann['segmentation']

        if mask.sum() < 5000:
            continue

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if not all(is_contour_smooth(contour, 100) for contour in contours):
            continue

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        mask_cropped = mask[rmin:rmax, cmin:cmax]
        image_cropped = original_image[rmin:rmax, cmin:cmax]

        output_image = np.zeros((rmax-rmin, cmax-cmin, 4), dtype=np.uint8)
        output_image[..., :3] = image_cropped
        output_image[..., 3] = mask_cropped * 255

        pil_image = Image.fromarray(output_image)
        pil_image_rgb = rgba_to_rgb_white_bg(pil_image)

        # Compute the hash of the current image
        current_hash = imagehash.phash(pil_image_rgb)

        # Check if a similar image has already been processed
        if any(current_hash - stored_hash <= 50 for stored_hash in image_hashes):
            # If similar, skip saving this image
            continue
        else:
            # If not similar, save the image and its hash
            image_hashes.add(current_hash)
            image_array.append(pil_image_rgb)
            # Optionally save the image to a file here

    return image_array
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


model_type = "vit_h"  #vit_b - 2:10 min, vit_l - 2:45 min, vit_h - 3:30 min
checkpoint_path = "CNNs/sam_vit_h_4b8939.pth" 

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