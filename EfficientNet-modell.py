from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

# 🚀 Sjekk om GPU er tilgjengelig
if tf.config.experimental.list_physical_devices('GPU'):
    print("✅ Using Metal GPU")
else:
    print("⚠️ Running on CPU")

# ✅ Global konfigurasjon
MODEL_NAME = 'efficientnet_model.h5'

BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.0001
TRAIN_DIR = 'data/'
PATIENCE = 5

# 🌟 Last inn eksisterende modell hvis tilgjengelig
def load_existing_model():
    if os.path.exists(MODEL_NAME):
        print(f"🔄 Loading existing model: {MODEL_NAME}")
        return load_model(MODEL_NAME)
    print("⚠️ No existing model found, building a new one.")
    return None

# 📦 Datahåndtering og augmentering
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

# 🧠 Bygg EfficientNet-modellen
def build_model(num_classes):
    model = Sequential([
        EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet', pooling='avg'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    print("✅ Model successfully built.")
    return model

# 🏋️‍♂️ Tren modellen
def train_model(model, train_data, val_data):
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=PATIENCE, restore_best_weights=True),
        ModelCheckpoint(MODEL_NAME, save_best_only=True, monitor='val_accuracy', mode='max')
    ]
    print("\n🚀 Starting training...")
    try:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        print("✅ Training completed successfully.")
        return history
    except Exception as e:
        print("❌ Error during training:", str(e))
        exit(1)

# 📊 Evaluering av modellen
def evaluate_model(model, val_data):
    print("\n📊 Evaluating the model on validation data...")
    try:
        loss, accuracy = model.evaluate(val_data)
        print(f"✅ Validation Loss: {loss:.4f} | Validation Accuracy: {accuracy:.4%}")
    except Exception as e:
        print("❌ Error during evaluation:", str(e))
        exit(1)

# 🚀 Generer trenings- og valideringsdata
train_data, val_data = create_data_generators()

print(f"Found {train_data.samples} training images in {len(train_data.class_indices)} classes.")
print(f"Found {val_data.samples} validation images in {len(val_data.class_indices)} classes.")

# 🧠 Bygg eller last inn modellen
model = load_existing_model()
if model is None:
    num_classes = len(train_data.class_indices)
    model = build_model(num_classes)

# 🏋️‍♂️ Start treningen
history = train_model(model, train_data, val_data)

# 📊 Evaluer modellen
evaluate_model(model, val_data)
