from segment_local import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import os
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
def is_contour_smooth(contour, threshold_angle):
    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]
        p3 = contour[(i + 2) % len(contour)][0]
        angle = np.abs(np.rad2deg(np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])))
        if angle < threshold_angle:
            return False
    return True

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def save_cropped_objects(masks, original_image, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    processed_boxes = []  # Store bounding boxes and areas
    image_array = []
    image_hashes = set()
    input_height, input_width = original_image.shape[:2]
    input_size = input_width * input_height

    for i, ann in enumerate(masks):
        mask = ann['segmentation']
        if mask.sum() < 5000:
            continue

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not all(is_contour_smooth(contour, 70) for contour in contours):
            continue

        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        rmin, rmax, cmin, cmax = np.where(rows)[0][0], np.where(rows)[0][-1], np.where(cols)[0][0], np.where(cols)[0][-1]

        mask_area = (rmax - rmin) * (cmax - cmin)
        if mask_area / input_size > 0.9:
            continue  # Skip masks nearly the size of the input image

        new_box = [cmin, rmin, cmax, rmax]

        # Check for significant overlap with larger processed masks
        skip_mask = False
        for processed_box in processed_boxes:
            if iou(new_box, processed_box[0]) > 0.1 and mask_area < processed_box[1] * 0.9:
                skip_mask = True
                break  # Skip smaller masks that are mostly covered by a larger mask

        if skip_mask:
            continue

        processed_boxes.append((new_box, mask_area))
        image_cropped = original_image[rmin:rmax, cmin:cmax]
        output_image = np.zeros((rmax-rmin, cmax-cmin, 4), dtype=np.uint8)
        output_image[..., :3] = image_cropped
        output_image[..., 3] = mask[rmin:rmax, cmin:cmax] * 255

        pil_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGRA2RGBA))
        pil_image_rgb = pil_image.convert("RGB")
        current_hash = imagehash.phash(pil_image_rgb)
        if any(current_hash - stored_hash <= 20 for stored_hash in image_hashes):
            continue
        image_hashes.add(current_hash)
        image_array.append(pil_image_rgb)
        pil_image_rgb.save(os.path.join(output_folder, f"object_{i}.png"))
                           
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
    output_folder = "croppedobjects"
    imarray = save_cropped_objects(masks, image, output_folder)


    print("All objects saved as individual images.")
    return imarray