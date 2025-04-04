import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import random

# Configuration
IMAGE_SIZE = (480, 480)  # Resize all images to 64x64
DATASET_PATH = 'data'  # Root dataset folder

# PLU mapping
plu_mapping = {
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
    '7622210410337': 'Kvikk Lunsj 3x47g Freia'
}

# Image augmentation function for contrast adjustment
def augment_image(image):
    # Contrast adjustment
    alpha = random.uniform(0.5, 1.5)  # Random contrast factor
    adjusted = cv2.convertScaleAbs(image, alpha=alpha)
    return adjusted

# Load images and labels
def load_images(dataset_path):
    X = []
    y = []
    
    # List all label folders
    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)
        
        # Check if it's a directory (i.e., a class folder)
        if not os.path.isdir(folder_path):
            continue  
        
        # Read all images (ignore _bb.png and .txt files)
        for filename in os.listdir(folder_path):
            if filename.endswith('.png') and '_bb' not in filename:  
                file_path = os.path.join(folder_path, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                if img is None:
                    continue
                
                # Apply contrast adjustment
                img = augment_image(img)
                
                img = cv2.resize(img, IMAGE_SIZE)  # Resize
                X.append(img.flatten())  # Flatten to 1D
                y.append(label)  # Folder name is the label

    return np.array(X), np.array(y)

# Load data
X, y = load_images(DATASET_PATH)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = SVC(kernel='linear')  # Support Vector Machine classifier
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.flatten().reshape(1, -1)  # Prepare for prediction
    
    prediction = clf.predict(img)
    return prediction[0]
a
# Example usage
test_image = 'test.png'
predicted_label = predict_image(test_image)

# Translate the predicted label using the PLU mapping
product_name = plu_mapping.get(predicted_label, "Unknown Product")
print("Predicted class:", product_name)