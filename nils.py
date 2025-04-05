from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from pathlib import Path
import pandas as pd

# 💾 Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("No GPU detected, using CPU.")

BATCH_SIZE = 64
EPOCHS = 2
IMG_SIZE = (128, 128)
LEARNING_RATE = 0.005
TRAIN_DIR = 'train'
VAL_DIR = 'val'
MODEL_PATH = 'efficientnet_model.h5'

# 🚀 Data preparation and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# 🧠 Build the model
model = Sequential([
    MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet'),
    GlobalAveragePooling2D(),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(len(train_data.class_indices), activation='softmax', dtype='float32')
])
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋️‍♂️ Train the model
with tf.device('/GPU:0'):
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# 💾 Save the model
model.save(MODEL_PATH, save_format='h5')

# ➡️ Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('efficientnet_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ Saved as efficientnet_model.h5 and efficientnet_model.tflite")

# 📈 Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)

# 🔍 Predict function
def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_name = list(train_data.class_indices.keys())[predicted_class]
    return class_name

# 🗣️ Label translations
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

# 🔁 Predict each image in data_20, skipping _bb.png and .txt
DATA_20_DIR = 'new_data/data_20'
total = 0
correct = 0
wrong = 0
results = []

for label_folder in Path(DATA_20_DIR).iterdir():
    if label_folder.is_dir():
        actual_label = label_folder.name
        for img_path in label_folder.iterdir():
            if (
                not img_path.is_file() or
                img_path.suffix.lower() != '.png' or
                img_path.name.endswith('_bb.png')
            ):
                continue

            try:
                predicted_label = predict_image(str(img_path))
                
                # Translate PLUs
                actual_product_name = plu_mapping.get(actual_label, "Unknown Product")
                predicted_product_name = plu_mapping.get(predicted_label, "Unknown Product")

                # Count accuracy
                if actual_product_name == predicted_product_name:
                    correct += 1
                else:
                    wrong += 1
                total += 1

                results.append({
                    'image_path': str(img_path),
                    'actual_label': actual_label,
                    'predicted_label': predicted_label,
                    'actual_label_translated': actual_product_name,
                    'predicted_label_translated': predicted_product_name,
                    'correct': actual_product_name == predicted_product_name
                })

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

# 📄 Save results to CSV
df = pd.DataFrame(results)
df.to_csv('prediction_results.csv', index=False)

# 📊 Summary
print(" ")
print(f"✅ Correct: {correct}/{total}")
print(f"❌ Wrong: {wrong}/{total}")
print(f"🎯 Accuracy: {(correct/total)*100:.2f}%")
