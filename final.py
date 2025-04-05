import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Configuration
IMAGE_SIZE = (128, 128)
MODEL_PATH = 'my_trained_model.h5'  # Path to your saved model
PLU_MAPPING = {
    '4011': 'Bananer Bama',
    '4015': 'Epler Røde',
    '4088': 'Paprika Rød',
    '4196': 'Appelsin',
    '94011': 'Bananer Økologisk',
    '90433917': 'Red Bull Regular 250ml boks',
    '90433924': 'Red Bull Sukkerfri 250ml boks',
    '7020097009819': 'Karbonadedeig 5% u/Salt og Vann 400g Meny',
    '7020097026113': 'Kjøttdeig Angus 14% 400g Meny',
    '7023026089401': 'Ruccula 65g Grønn&Frisk',
    '7035620058776': 'Rundstykker Grove Fullkorn m/Frø Rustikk 6stk 420g',
    '7037203626563': 'Leverpostei Ovnsbakt Orginal 190g Gilde',
    '7037206100022': 'Kokt Skinke Ekte 110g Gilde',
    '7038010009457': 'Yoghurt Skogsbær 4x150g Tine',
    '7038010013966': 'Norvegia 26% skivet 150g Tine',
    '7038010021145': 'Jarlsberg 27% skivet 120g Tine',
    '7038010054488': 'Cottage Cheese Mager 2% 400g Tine',
    '7038010068980': 'Yt Protein Yoghurt Vanilje 430g Tine',
    '7039610000318': 'Frokostegg Frittgående L 12stk Prior',
    '7040513000022': 'Gulrot 750g Beger',
    '7040513001753': 'Gulrot 1kg pose First Price',
    '7040913336684': 'Evergood Classic Filtermalt 250g',
    '7044610874661': 'Pepsi Max 0,5l flaske',
    '7048840205868': 'Frokostyoghurt Skogsbær 125g pose Q',
    '7071688004713': 'Original Havsalt 190g Sørlandschips',
    '7622210410337': 'Kvikk Lunsj 3x47g Freia',
    '69': 'Empty'
}

# Load the saved model
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from {MODEL_PATH}")

# Manually define class indices (you should have this from your training script)
class_indices = {
    0: '4011',
    1: '4015',
    2: '4088',
    3: '4196',
    4: '94011',
    5: '90433917',
    6: '90433924',
    7: '7020097009819',
    8: '7020097026113',
    9: '7023026089401',
    10: '7035620058776',
    11: '7037203626563',
    12: '7037206100022',
    13: '7038010009457',
    14: '7038010013966',
    15: '7038010021145',
    16: '7038010054488',
    17: '7038010068980',
    18: '7039610000318',
    19: '7040513000022',
    20: '7040513001753',
    21: '7040913336684',
    22: '7044610874661',
    23: '7048840205868',
    24: '7071688004713',
    25: '7622210410337',
    26: '69'  # Example empty class
}

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

# Predict the product based on the image
def predict_product(image_path):
    img_array = load_and_preprocess_image(image_path)
    
    # Predict using the model
    prediction = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    # Get the PLU from the class_indices
    predicted_plu = class_indices[predicted_class_index]
    
    # Get the product name using the PLU mapping
    predicted_product_name = PLU_MAPPING.get(predicted_plu, "Unknown Product")
    
    return predicted_product_name, predicted_plu

# Function to check if a file with the same name exists and append a number to make it unique
def get_unique_filename(folder_path, base_name):
    counter = 1
    filename = base_name
    while os.path.exists(os.path.join(folder_path, filename)):
        name, ext = os.path.splitext(base_name)
        filename = f"{name}_{counter}{ext}"
        counter += 1
    return filename

# Process all images in the folder and rename them
def process_and_rename_images(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image (you can add other image formats here if needed)
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            print(f"Processing {filename}...")
            
            # Predict the product name
            product_name, plu_code = predict_product(file_path)
            
            # Sanitize product name for file system (replace spaces with underscores, remove invalid characters)
            sanitized_name = product_name.replace(" ", "_").replace("/", "_") + ".jpg"
            
            # Ensure unique filename (if a file with the same name exists, add a number)
            unique_name = get_unique_filename(folder_path, sanitized_name)
            
            # Define the new file path
            new_file_path = os.path.join(folder_path, unique_name)
            
            # Rename the image file
            os.rename(file_path, new_file_path)
            print(f"Renamed to: {unique_name}")

# Example usage
folder_path = 'cropped_screenshots'  # Replace with the path to your folder
process_and_rename_images(folder_path)
