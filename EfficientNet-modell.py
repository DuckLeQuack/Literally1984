from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 1. Datagenerator for dataforsterkning
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% til validering
)

# 2. Last inn trenings- og valideringsdata
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

# 3. Bygg modellen
model = Sequential([
    EfficientNetB0(input_shape=(224, 224, 3), include_top=False, weights='imagenet'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')  # Dynamisk antall klasser
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Tren modellen
print("\n🚀 Starter trening...\n")
history = model.fit(train_data, validation_data=val_data, epochs=10)

# 5. Evaluering av modellen
print("\n📊 Evaluering av modellen...\n")
loss, accuracy = model.evaluate(val_data)
print(f"Valideringsnøyaktighet: {accuracy * 100:.2f}%")

# 6. Lagre modellen
model.save('efficientnet_model.h5')
print("\n💾 Modell lagret som efficientnet_model.h5\n")

# 7. Test med et nytt bilde
def predict_image(image_path):
    # Last opp og forbered bildet
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Forutsi klassen
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_name = list(train_data.class_indices.keys())[predicted_class]

    # Skriv ut resultatet
    print(f"📷 Predikert klasse: {class_name}")
    return class_name

# Test med et eksempelbilde
example_image = 'data/paprika/img1.jpg'  # Oppdater med riktig sti
predict_image(example_image)

# 8. Visualisere treningshistorikken
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Nøyaktighet
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Modellnøyaktighet')
    plt.xlabel('Epoker')
    plt.ylabel('Nøyaktighet')
    plt.legend()

    # Tap
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Modelltap')
    plt.xlabel('Epoker')
    plt.ylabel('Tap')
    plt.legend()

    plt.show()

plot_training_history(history)

# 9. Visualisere et eksempelbilde fra treningsdata
def show_sample_image(data):
    images, labels = next(data)
    plt.imshow(images[0])
    predicted_label = list(train_data.class_indices.keys())[np.argmax(labels[0])]
    plt.title(f"Etikett: {predicted_label}")
    plt.axis('off')
    plt.show()

show_sample_image(train_data)


# Test med et nytt bilde etter trening
example_image = 'data/paprika/img1.jpg'  # Oppdater med riktig sti
predict_image(example_image)

# Eller direkte kall med funksjonen
predict_image('data/banan/banana1.jpg')