import os
import cv2
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0001
TRAIN_DIR = 'new_data/data_80'
TEST_DIR = 'new_data/data_20'# Check for GPU availability and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(":white_tick: Using GPU")
    except RuntimeError as e:
        print(f":x: Error setting GPU: {e}")
else:
    print(":warning: Running on CPU")# Data Augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    channel_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='reflect'
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)# Optimized CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),    tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])# Training
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)# Evaluation
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes
print(classification_report(y_true, y_pred_classes, target_names=list(val_data.class_indices.keys())))

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