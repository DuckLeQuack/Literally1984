from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Datagenerator for dataforsterkning
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Bygge modellen
model = Sequential([
    EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='softmax')  # 26 kategorier
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Trening
model.fit(train_data, validation_data=val_data, epochs=10)
