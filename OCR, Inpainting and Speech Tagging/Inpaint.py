import cv2
import numpy as np
import easyocr
import os
import json

def detect_text_and_inpaint(image_path):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Load the image
    img = cv2.imread(image_path)

    mask = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)[1][:, :, 0]
    
    new_mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    
    dst = cv2.inpaint(img, new_mask, 5, cv2.INPAINT_TELEA)
    
    return dst

# Function to process images specified in JSONL file
def process_images_from_jsonl(jsonl_file, img_folder):
    # Extract the filename without extension
    filename = os.path.splitext(os.path.basename(jsonl_file))[0]

    # Output folder for inpainted images
    output_folder = f'inpainted_images_{filename}'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read JSONL file and process each image
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            img_path = data['img']
            output_image_path = os.path.join(output_folder, os.path.basename(data['img']))
            if os.path.exists(img_path):
                inpainted_image = detect_text_and_inpaint(img_path)
                cv2.imwrite(output_image_path, inpainted_image)

# Specify the JSONL file and the folder containing images
jsonl_file = 'train.jsonl'
img_folder = 'img'

# Call the function
process_images_from_jsonl(jsonl_file, img_folder)

print("Text inpainting completed. Inpainted images saved in the corresponding folder.")
