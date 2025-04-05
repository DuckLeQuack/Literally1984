import cv2
import os

# Function to crop the center of an image based on width and from the bottom for the height
def crop_center(image, target_width=285, target_height=270):
    height, width = image.shape[:2]
    
    # Crop width based on the center
    crop_x = (width - target_width) // 2
    
    # Crop height to 270 pixels from the bottom
    crop_y = height - target_height
    
    return image[crop_y:height, crop_x:crop_x+target_width]

# Folder containing the screenshots
input_folder = 'screenshots'
output_folder = 'cropped_screenshots'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through each image in the screenshots folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):  # Only process PNG files
        input_image_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_image_path)

        if image is not None:
            # Crop the image with the modified logic
            cropped_image = crop_center(image)

            # Save the cropped image to the output folder
            output_image_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Cropped image saved: {output_image_path}")
        else:
            print(f"Failed to load image: {input_image_path}")
