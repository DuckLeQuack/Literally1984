import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Configuration
IMAGE_SIZE = (64, 64)  # Resize all images to 64x64
DATASET_PATH = 'data'  # Root dataset folder

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

# Example usage
test_image = 'test.png'
predicted_label = predict_image(test_image)
print("Predicted class:", predicted_label)
