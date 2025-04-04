import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Configuration
IMAGE_SIZE = (64, 64)  # CNNs work better with slightly larger images
DATASET_PATH = 'data'

# Load images and labels
def load_images(dataset_path):
    X = []
    y = []
    class_labels = {}  # Store label-to-index mapping
    label_index = 0
    
    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue  
        
        # Assign numerical label
        if label not in class_labels:
            class_labels[label] = label_index
            label_index += 1

        for filename in os.listdir(folder_path):
            if filename.endswith('.png') and '_bb' not in filename:  
                file_path = os.path.join(folder_path, filename)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  
                if img is None:
                    continue
                img = cv2.resize(img, IMAGE_SIZE)  
                X.append(img)
                y.append(class_labels[label])  # Convert folder name to numerical label

    return np.array(X), np.array(y), class_labels

# Load data
X, y, class_labels = load_images(DATASET_PATH)

# Normalize images (0 to 1 range)
X = X / 255.0  

# Reshape for CNN input (batch, height, width, channels)
X = X.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(class_labels))

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Prevent overfitting
    Dense(len(class_labels), activation='softmax')  # Output layer for classification
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Save model (optional)
model.save("cnn_model.h5")

# Evaluate performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Predict a single image
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0  # Normalize
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)  # Reshape for CNN

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get index of highest confidence
    label_name = [k for k, v in class_labels.items() if v == predicted_class][0]
    
    return label_name

# Test with an image
test_image = 'test.png'
predicted_label = predict_image(test_image)
print("Predicted class:", predicted_label)

