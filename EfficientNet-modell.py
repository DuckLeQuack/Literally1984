from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# 🚀 Check GPU or CPU availability
if tf.config.experimental.list_physical_devices('GPU'):
    print("✅ Using Metal GPU")
else:
    print("⚠️ Running on CPU")

# ✅ Global Configuration
MODEL_NAME = 'efficientnet_model.h5'
BATCH_SIZE = 16
EPOCHS = 10
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.0001
TRAIN_DIR = 'data/'
PATIENCE = 5

# 🌟 Load an existing model if available
def load_existing_model():
    if os.path.exists(MODEL_NAME):
        print(f"🔄 Loading existing model: {MODEL_NAME}")
        return load_model(MODEL_NAME)
    print("⚠️ No existing model found, building a new one.")
    return None

# 📦 Data Augmentation and Loading
def create_data_generators():
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    print("🔄 Loading training data...")
    try:
        train_data = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training'
        )
        val_data = datagen.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation'
        )
        if len(train_data) == 0 or len(val_data) == 0:
            print("❌ No images found. Check your data folder structure.")
            exit(1)
        print("✅ Data successfully loaded.")
        return train_data, val_data
    except Exception as e:
        print("❌ Error loading data:", str(e))
        exit(1)

# 🧠 Build the EfficientNet Model
def build_model(num_classes):
    model = Sequential([
        EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 🏋️‍♂️ Train the Model
def train_model(model, train_data, val_data):
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy', mode='max')
    ]
    print("\n🚀 Starting training...")
    try:
        history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks, verbose=1)
        print("✅ Training completed successfully.")
        return history
    except Exception as e:
        print("❌ Error during training:", str(e))
        exit(1)
