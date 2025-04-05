import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Configuration
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.0001
TRAIN_DIR = 'new_data/data_80'
TEST_DIR = 'new_data/data_20'

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

# Set GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Using GPU")
    except RuntimeError as e:
        print(f"❌ Error setting GPU: {e}")
else:
    print("⚠️ Running on CPU")

# Data Augmentation
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

# Data loading
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
    class_mode='categorical',
    shuffle=False  # Important for consistent evaluation
)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Predict
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=list(val_data.class_indices.keys())))

# Translate predicted and actual class labels to PLUs
index_to_label = {v: k for k, v in val_data.class_indices.items()}  # reverse mapping

total = 0
correct = 0
wrong = 0
for i in range(len(y_true)):
    actual_label = index_to_label[y_true[i]]
    predicted_label = index_to_label[y_pred_classes[i]]
    
    actual_product_name = plu_mapping.get(actual_label, "Unknown Product")
    predicted_product_name = plu_mapping.get(predicted_label, "Unknown Product")
    
    if actual_product_name == predicted_product_name:
        correct += 1
    else:
        wrong += 1
    total += 1

print("\n🧠 PLU Name Matching Accuracy:")
print(f"✅ Correct: {correct}/{total}")
print(f"❌ Wrong: {wrong}/{total}")
