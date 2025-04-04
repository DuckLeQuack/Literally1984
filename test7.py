import os
import cv2
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuration
IMAGE_SIZE = (64, 64)  # Resize all images to 64x64
DATASET_PATH = 'data'  # Root dataset folder

# Check for GPU availability and set memory growth for TensorFlow
gpus = tf.config.list_physical_devices('GPU')  # List GPUs available
if gpus:
    try:
        # Set memory growth for each GPU to allow dynamic memory allocation
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Set memory growth
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU detected, using CPU.")

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
                X.append(img)  # Keep as 2D array
                y.append(label)  # Folder name is the label

    return np.array(X), np.array(y)

# Load data
X, y = load_images(DATASET_PATH)

# Normalize pixel values to [0, 1]
X = X.astype('float32') / 255.0

# Convert labels to integers
label_to_index = {label: idx for idx, label in enumerate(set(y))}
y = np.array([label_to_index[label] for label in y])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)  # Add channel dimension
X_test = X_test.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Build a simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_to_index), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with GPU acceleration (if available)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32') / 255.0  # Normalize
    img = img.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)  # Reshape for prediction
    
    # Ensure prediction uses GPU if available
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Example usage
test_image = 'test.png'
predicted_label_index = predict_image(test_image)

# Translate the predicted label using the PLU mapping
product_class = list(label_to_index.keys())[list(label_to_index.values()).index(predicted_label_index)]

product_name = plu_mapping[f"{product_class}"]

# Print the predicted PLU and product name
print(f"Predicted class: {product_class}")
print(f"Name of Product: {product_name}")
