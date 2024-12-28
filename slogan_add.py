import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

def find_free_space(image, min_contour_area=5000):
    # Convert PIL Image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Thresholding to create a binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Invert the binary image to get free spaces
    free_space = cv2.bitwise_not(binary)

    # Find contours of the free spaces
    contours, _ = cv2.findContours(free_space, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the free space
    mask = np.zeros_like(free_space)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return mask

def add_slogan_with_detection(image, slogan, font_path, font_size=48, font_color="black"):
    free_space_mask = find_free_space(image)

    # Convert the image to use PIL for drawing
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Get text size using textbbox
    bbox = draw.textbbox((0, 0), slogan, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Search for a suitable position within the free space
    for y in range(free_space_mask.shape[0] - text_height):
        for x in range(free_space_mask.shape[1] - text_width):
            # Check if the area is free (white in the mask)
            if np.all(free_space_mask[y:y + text_height, x:x + text_width] == 255):
                # Use the font color specified by the parameter
                draw.text((x, y), slogan, fill=font_color, font=font)
                
                # Save image to bytes buffer
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                return img_byte_arr  

    print("No suitable space found for the slogan.")
    return None  

def generate_image_with_slogan(image, slogan, font_path):
    font_color = "white"
    result = add_slogan_with_detection(image, slogan, font_path, font_color=font_color)
    if result is None:
        print("No space found for slogan.")
    return result  
