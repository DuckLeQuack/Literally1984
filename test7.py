import os
import cv2
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Configuration
IMAGE_SIZE = (240, 240)  # Resize all images to 64x64
TRAINING_DATA_PATH = 'new_data/data_80'  # Root training dataset folder
TESTING_DATA_PATH = 'new_data/data_20'  # Root testing dataset folder

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
                
                # Resize the cropped image to IMAGE_SIZE (240x240)
                img = cv2.resize(img, IMAGE_SIZE)  
                X.append(img)  # Keep as 2D array
                y.append(label)  # Folder name is the label

    return np.array(X), np.array(y)

# Load training data
X_train, y_train = load_images(TRAINING_DATA_PATH)

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0

# Convert labels to integers
label_to_index = {label: idx for idx, label in enumerate(set(y_train))}
y_train = np.array([label_to_index[label] for label in y_train])

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)  # Add channel dimension
X_val = X_val.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

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
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

# Evaluate the model on testing data (data_20)
X_test, y_test = load_images(TESTING_DATA_PATH)

# Normalize the test images
X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# Convert the test labels to integers
y_test = np.array([label_to_index[label] for label in y_test])

# Predict and evaluate results
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_test, y_pred_classes))

# Check and print predicted vs actual labels

total = 0
correct = 0
wrong = 0
for i in range(len(X_test)):
    actual_label = list(label_to_index.keys())[list(label_to_index.values()).index(y_test[i])]
    predicted_label_idx = y_pred_classes[i]
    predicted_label = list(label_to_index.keys())[list(label_to_index.values()).index(predicted_label_idx)]
    
    # Translate using PLU mapping
    actual_product_name = plu_mapping.get(actual_label, "Unknown Product")
    predicted_product_name = plu_mapping.get(predicted_label, "Unknown Product")
    
    #print(f"Image {i + 1}:")
    #print(f"  Actual Label: {actual_product_name} (PLU: {actual_label})")
    #print(f"  Predicted Label: {predicted_product_name} (PLU: {predicted_label})")
    print()
    if actual_product_name == predicted_product_name:
        correct += 1
    else:
        wrong += 1
    total += 1

print(" ")
print(f"Correct: {correct}/{total}")
print(f"Wrong: {wrong}/{total}")
